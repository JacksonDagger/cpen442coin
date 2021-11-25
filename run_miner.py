from os import error
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

def difficulty_pass(preceeding, blob):
    hash = cpen442coinhash(preceeding, blob)
    difficulty = 0
    for i in range(length(hash)):
        if hash[i] == '0':
            difficulty += 1
        else:
            break
    return difficulty

def main():
    log = {}
    log["hashrate"] = 0
    log["coinrate"] = 0
    log["coinsmined"] = 0

    logfileName = "logs/cpen442coinrun" + datetime.now().strftime("%Y-%d-%m--%H-%M-%S") + ".log"
    coinlogfileName = "logs/cpen442coinmines" + datetime.now().strftime("%Y-%d-%m--%H-%M-%S") + ".log"
    
    lastCoin = "00000000002dee43c5ded98ccf60d2e7981030d96091325844b0b9d29e8e4278"
    difficulty = 11

    while(True):
        try:
            lastCoinResponseStr = requests.post("http://cpen442coin.ece.ubc.ca/last_coin")
            lastCoin = lastCoinResponseStr.json()["coin_id"]

            difficultyResponseStr = requests.post("http://cpen442coin.ece.ubc.ca/difficulty")
            difficulty = int(difficultyResponseStr.json()["number_of_leading_zeros"])
        except:
            pass

        args = ["./parallel_miner", lastCoin, str(difficulty)]
        output = subprocess.run(args, capture_output=True)
        log["lastcoin"] = lastCoin
        log["difficulty"] = difficulty
        ret = output.stdout.decode()

        if "success:" in ret:
            blob = ret[len("success:"):]
            b64 = b64encode(bytes.fromhex(blob)).decode()
            data = {
            "coin_blob":b64,
            "id_of_miner":miner_id
            }
            response = requests.post('http://cpen442coin.ece.ubc.ca/claim_coin', data = data)
            data["code"] = response.status_code
            log["coinsmined"] += 1
            data["coinsmined"] = log["coinsmined"]

            with open(coinlogfileName, "a") as coinlogfile:
                coinlogfile.write(json.dumps(data))
                coinlogfile.write("\n")
        else:
            log["hashrate"] = float(ret.split()[-1])
            log["coinrate"] = 60*60*log["hashrate"]/(2**(4*difficulty))
        
        with open(logfileName, "a") as logfile:
            log["timestamp"] = datetime.now().strftime("%Y-%d-%m--%H:%M:%S")
            logfile.write(json.dumps(log))
            logfile.write("\n")

        outstr = "coinrate: " + str(round(log["coinrate"], 4)) + ", coinsmined: " + str(log["coinsmined"]) + ", difficulty: " + str(log["difficulty"])
        sys.stdout.flush()
        print(outstr, end='\r')

if __name__ == "__main__":
    # cpen442coinhash("0000000000b6c5adad1665131871458fc1217306af9e79153a42eb4ed4b6b7b6", "7089F34E1D29610B")
    main()