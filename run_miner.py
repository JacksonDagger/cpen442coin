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

    logdir = "logs"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    log = {}
    log["hashrate"] = 0
    log["coinrate"] = 0
    log["predcoins"] = 0
    log["coinsmined"] = 9
    log["lastCoin"] = "000000000039f30c6e714e3e4551f91cebea6cbbf42a0cbc0df4ea5d2c48debc"
    log["difficulty"] = 10
    log["foundBlob"] = ""
    log["foundDif"] = 0
    log["lastCoinTime"] = 0

    period = 0
    difDrop = 600*60

    try:
        with open(stateFilename, "r") as stateFile:
            statelog = json.load(stateFile)
            for key in statelog.keys():
                log[key] = statelog[key]
    except Exception as e:
        logEvent(dumpfileName, "json load error", str(e))

    contestEnd = datetime(2021, 12, 1, 21, 0, 0)
    
    req_interval = 1
    req_num = 0

    while(True):
        req_num += 1
        with open(logfileName, "a") as logfile:
            log["timestamp"] = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
            logfile.write(json.dumps(log))
            logfile.write("\n")

        oldCoin = log["lastCoin"]
        if not (req_num % req_interval):
            try:
                lastCoinResponseStr = requests.post("http://cpen442coin.ece.ubc.ca/last_coin")
                log["lastCoin"] = lastCoinResponseStr.json()["coin_id"]
                log["lastCoinTime"] = lastCoinResponseStr.json()["time_stamp"]

                difficultyResponseStr = requests.post("http://cpen442coin.ece.ubc.ca/difficulty")
                log["difficulty"] = int(difficultyResponseStr.json()["number_of_leading_zeros"])
            except Exception as e:
                logEvent(dumpfileName, "server error", str(e) + " req interval:" + str(req_interval))
                req_interval *= 2
                pass

        period = 30*60 if log["difficulty"] >= 10 else 0
        
        if log["lastCoin"] != oldCoin:
            logEvent(dumpfileName, "newcoin", "old: " + oldCoin + ", new: " + log["lastCoin"])
            log["foundBlob"] = ""
            log["foundDif"] = 0
            
        if log["foundDif"] >= log["difficulty"]:
            b64 = b64encode(bytes.fromhex(log["foundBlob"])).decode()
            data = {
            "coin_blob":b64,
            "id_of_miner":miner_id
            }

            wait = period - (datetime.now().timestamp() - log["lastCoinTime"])
            if wait > 60:
                wait = 60
            if wait > 0:
                logEvent(dumpfileName, "close interval sleep", str(wait))
                print("sleeping 60s")
                sleep(wait)

            response = requests.post('http://cpen442coin.ece.ubc.ca/claim_coin', json = data)
            log["coinsmined"] += 1

            data["code"] = response.status_code
            if data["code"] == 200:
                log["coinsmined"] += 1
            else:
                logEvent(dumpfileName, "bad response", str(response))
            data["coinsmined"] = log["coinsmined"]
            data["timestamp"] = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")

            with open(coinlogfileName, "a") as coinlogfile:
                coinlogfile.write(json.dumps(data))
                coinlogfile.write("\n")
                
            continue

        elif log["foundDif"] >= 10 and log["difficulty"] >= 11:
            wait = int(log["lastCoinTime"] + difDrop - datetime.now().timestamp())
            partwait = wait
            if partwait > 122:
                partwait = 122
            partwait -= 2
            if wait < 240:
                if partwait > 0:
                    logEvent(dumpfileName, "high dif sleep ", "total: " + str(wait) + ", interval: " + str(partwait))
                    print("sleeping " + str(partwait) + " with total expected wait of: " + str(wait))
                    sleep(partwait)
                if wait >= -60*60:
                    continue

        args = [binpath, log["lastCoin"]]
        output = subprocess.run(args, capture_output=True)
        ret = output.stdout.decode()

        if "success:" in ret:
            newFoundBlob = ret[len("success:"):]
            newFoundDif = difficulty_check(log["lastCoin"], newFoundBlob)

            if newFoundDif > log["foundDif"]:
                log["foundDif"] = newFoundDif
                log["foundBlob"] = newFoundBlob
            
            if newFoundDif < 9:
                blobLog = log
                blobLog["hash"] = cpen442coinhash(log["lastCoin"], log["foundBlob"])
                logEvent(dumpfileName, "bad blob", str(blobLog))
        else:
            log["hashrate"] = float(ret.split()[-1])
            log["coinrate"] = 60*60*log["hashrate"]/(2**(4*log["difficulty"]))
            hoursLeft = (contestEnd - datetime.now()) / timedelta(hours=1)
            log["predcoins"] = hoursLeft*log["coinrate"]

        logEvent(dumpfileName, "stdout", ret)
        printLogStr(log)
        
        try:
            with open(stateFilename, "w") as stateFile:
                json.dump(log, stateFile)
        except Exception as e:
            logEvent(dumpfileName, "error saving state", str(e))


if __name__ == "__main__":
    main()