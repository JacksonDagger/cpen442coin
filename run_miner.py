import os
import requests
from base64 import b64encode, b64decode
import subprocess
from datetime import datetime, timedelta
from time import sleep
import json
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

def strRound(num, digits):
    formatStr = "{:0." + str(digits) + "f}"
    rNum = round(num, digits)
    return formatStr.format(rNum)

def printLogStr(log):
    outstr = "hashrate (Mh/s):" +strRound(log["hashrate"]/1000000, 2) + \
        ", coinrate (c/h):" + strRound(log["coinrate"], 4) + \
        ", predcoins:" + strRound(log["predcoins"], (2 if log["coinrate"] < 0.2 else 1)) + \
        ", coinsmined:" + str(log["coinsmined"]) + \
        ", dif:" + str(log["difficulty"]) + \
        ", fdif:" + str(log["foundDif"])
    #sys.stdout.flush()
    print(outstr) #, end='\r')

def logEvent(dumpfileName, eventType, details):
    eventDetails = {}
    eventDetails["timestamp"] = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    eventDetails["eventType"] = eventType
    eventDetails["details"] = details

    with open(dumpfileName, "a") as dumpfile:
        dumpfile.write(json.dumps(eventDetails))
        dumpfile.write("\n")


def main():
    binpath = "./bin/gpuminer-hip"
    logfileName = "logs/coin442tries" + datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + ".log"
    coinlogfileName = "logs/coin442mines" + datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + ".log"
    dumpfileName = "logs/logdump" + datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + ".log"
    stateFilename = "logs/state.json"

    log = {}
    log["hashrate"] = 0
    log["coinrate"] = 0
    log["predcoins"] = 0

    lastCoin = "000000000039f30c6e714e3e4551f91cebea6cbbf42a0cbc0df4ea5d2c48debc"
    evenDif = 8
    difficulty = 10
    foundBlob = ""
    foundDif = 0
    lastCoinTime = 0
    period = 0

    log["coinsmined"] = 7

    try:
        with open(stateFilename, "r") as stateFile:
            statelog = json.load(stateFile)
            for key in statelog.keys():
                log[key] = statelog[key]
            lastCoin = log["lastcoin"]
            foundBlob = log["foundBlob"]
            foundDif = log["foundDif"]
            # lastCoinTime = log["l"]
    except Exception as e:
        logEvent(dumpfileName, "json load error", str(e))

    logdir = "logs"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    contestEnd = datetime(2021, 12, 1, 21, 0, 0)
    

    req_interval = 1
    req_num = 0

    while(True):
        oldCoin = lastCoin
        if not (req_num % req_interval):
            try:
                lastCoinResponseStr = requests.post("http://cpen442coin.ece.ubc.ca/last_coin")
                lastCoin = lastCoinResponseStr.json()["coin_id"]
                lastCoinTime = lastCoinResponseStr.json()["time_stamp"]

                difficultyResponseStr = requests.post("http://cpen442coin.ece.ubc.ca/difficulty")
                difficulty = int(difficultyResponseStr.json()["number_of_leading_zeros"])
            except Exception as e:
                logEvent(dumpfileName, "server error", str(e) + " req interval:" + str(req_interval))
                req_interval *= 2
                pass

        if lastCoin != oldCoin:
            evenDif = difficulty - difficulty % 2
            foundBlob = ""
            foundDif = 0
            
        if foundDif >= difficulty:
            b64 = b64encode(bytes.fromhex(foundBlob)).decode()
            data = {
            "coin_blob":b64,
            "id_of_miner":miner_id
            }

            wait = period - (datetime.now().timestamp() - lastCoinTime)
            if wait > 0:
                #logEvent(dumpfileName, "sleep", str(wait))
                #sleep(wait)
                pass
            response = requests.post('http://cpen442coin.ece.ubc.ca/claim_coin', json = data)
            
            log["coinsmined"] += 1

            data["code"] = response.status_code
            data["coinsmined"] = log["coinsmined"]
            data["timestamp"] = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")

            with open(coinlogfileName, "a") as coinlogfile:
                coinlogfile.write(json.dumps(data))
                coinlogfile.write("\n")

        log["lastcoin"] = lastCoin
        log["difficulty"] = difficulty
        log["evenDif"] = evenDif

        # args = [binpath, lastCoin, str(evenDif)]
        args = [binpath, lastCoin]
        output = subprocess.run(args, capture_output=True)
        ret = output.stdout.decode()

        if "success:" in ret:
            newFoundBlob = ret[len("success:"):]
            newFoundDif = difficulty_check(lastCoin, newFoundBlob)

            if newFoundDif > foundDif:
                foundDif = newFoundDif
                foundBlob = newFoundBlob
            
            if newFoundDif < 9: # evenDif:
                blobLog = log
                blobLog["foundBlob"] = foundBlob
                blobLog["foundDif"] = foundDif
                blobLog["hash"] = cpen442coinhash(lastCoin, foundBlob)
                logEvent(dumpfileName, "bad blob", str(blobLog))
        else:
            log["hashrate"] = float(ret.split()[-1])
            log["coinrate"] = 60*60*log["hashrate"]/(2**(4*difficulty))
            hoursLeft = (contestEnd - datetime.now()) / timedelta(hours=1)
            log["predcoins"] = hoursLeft*log["coinrate"]

        log["foundBlob"] = foundBlob
        log["foundDif"] = foundDif

        with open(logfileName, "a") as logfile:
            log["timestamp"] = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
            logfile.write(json.dumps(log))
            logfile.write("\n")

        logEvent(dumpfileName, "stdout", ret)
        printLogStr(log)
        req_num += 1
        try:
            with open(stateFilename, "w") as stateFile:
                json.dump(log, stateFile)
        except Exception as e:
            logEvent(dumpfileName, "error saving state", str(e))


if __name__ == "__main__":
    main()