import os
import requests
from base64 import b64encode, b64decode
import subprocess
from datetime import datetime
import json
import sys
from Crypto.Hash import SHA256

miner_id = "e8d2dadb3c5ead451b8943cff5ef909ef0f3de313c4d274d85d2d3c8a5a30c1f"

def cpen442coinhash(preceeding, blob):
    h = SHA256.new()
    h.update(b'CPEN 442 Coin')
    h.update(b'2021')
    h.update(preceeding.encode())
    h.update(bytes.fromhex(blob))
    h.update(miner_id.encode())
    return h.hexdigest()

def difficulty_check(preceeding, blob):
    hash = cpen442coinhash(preceeding, blob)
    difficulty = 0
    for i in range(len(hash)):
        if hash[i] == '0':
            difficulty += 1
        else:
            break
    return difficulty

def send_coin(blob, coinlogfileName, coinsmined=0):
    b64 = b64encode(bytes.fromhex(blob)).decode()
    data = {
    "coin_blob":b64,
    "id_of_miner":miner_id
    }
    response = requests.post('http://cpen442coin.ece.ubc.ca/claim_coin', data = data)
    data["code"] = response.status_code
    data["coinsmined"] = coinsmined
    data["timestamp"] = datetime.now().strftime("%Y-%d-%m--%H:%M:%S")

    with open(coinlogfileName, "a") as coinlogfile:
        coinlogfile.write(json.dumps(data))
        coinlogfile.write("\n")

def main():
    binpath = "./bin/cpu_miner"

    log = {}
    log["hashrate"] = 0
    log["coinrate"] = 0
    log["coinsmined"] = 0

    logdir = "logs"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logfileName = "logs/cpen442coinrun" + datetime.now().strftime("%Y-%d-%m--%H-%M-%S") + ".log"
    coinlogfileName = "logs/coinsmined" + datetime.now().strftime("%Y-%d-%m--%H-%M-%S") + ".log"
    dumpfileName ="logs/stdoutdump" + datetime.now().strftime("%Y-%d-%m--%H-%M-%S") + ".log"

    lastCoin = "00000000002dee43c5ded98ccf60d2e7981030d96091325844b0b9d29e8e4278"
    incDifficulty = 8
    difficulty = 11
    foundblob = ""

    while(True):
        oldCoin = lastCoin
        try:
            lastCoinResponseStr = requests.post("http://cpen442coin.ece.ubc.ca/last_coin")
            lastCoin = lastCoinResponseStr.json()["coin_id"]

            difficultyResponseStr = requests.post("http://cpen442coin.ece.ubc.ca/difficulty")
            difficulty = int(difficultyResponseStr.json()["number_of_leading_zeros"])
        except:
            pass

        if lastCoin != oldCoin:
            incDifficulty = 8 if difficulty > 8 else difficulty
            foundblob = ""
        elif difficulty < incDifficulty:
            log["coinsmined"] += 1
            send_coin(foundblob, coinlogfileName, log["coinsmined"])

        log["lastcoin"] = lastCoin
        log["difficulty"] = difficulty
        log["incDifficulty"] = incDifficulty

        args = [binpath, lastCoin, str(incDifficulty)]
        output = subprocess.run(args, capture_output=True)
        ret = output.stdout.decode()

        if "success:" in ret:
            blob = ret[len("success:"):]
            difFound = difficulty_check(lastCoin, blob)
            incDifficulty = difFound + 1
            foundblob = blob
        else:
            log["hashrate"] = float(ret.split()[-1])
            log["coinrate"] = 60*60*log["hashrate"]/(2**(4*difficulty))
        
        with open(logfileName, "a") as logfile:
            log["timestamp"] = datetime.now().strftime("%Y-%d-%m--%H:%M:%S")
            logfile.write(json.dumps(log))
            logfile.write("\n")
        with open(dumpfileName, "a") as dumpfile:
            dumpfile.write(ret)
            dumpfile.write("\n")

        outstr = "coinrate: " + str(round(log["coinrate"], 4)) + ", coinsmined: " + str(log["coinsmined"]) + ", difficulty: " + str(log["difficulty"])
        sys.stdout.flush()
        print(outstr, end='\r')

if __name__ == "__main__":
    main()