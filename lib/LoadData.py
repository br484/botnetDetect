#!/usr/bin/python
# encoding=utf8

import socket, struct, sys, pickle, random
import numpy as np

def loaddata(fileName):
    
    file = open(fileName, 'r')
    it = 10000000;

    xdata = []
    ydata = []
    xdataT = []
    ydataT = []
    flag=0
    count1=0
    count2=0
    count3=0
    count4=0

    #determina a conversão de protocolos/estados em números inteiros
    protoDict = {'arp': 5, 'unas': 13, 'udp': 1, 'rtcp': 7, 'pim': 3, 'udt': 11, 'esp': 12, 'tcp' : 0, 'rarp': 14, 'ipv6-icmp': 9, 'rtp': 2, 'ipv6': 10, 'ipx/spx': 6, 'icmp': 4, 'igmp' : 8}

    file.readline()
    x = 30
    for line in file:
        sd = line[:-1].split(',')
        dur, Sip, Dip, proto, totB, Sport, Dport, label = sd[0],sd[1], sd[2], sd[3], sd[4], sd[5], sd[6], sd[7]
        proto = proto.upper();
        try:
            Sip = socket.inet_aton(Sip)
            Sip = struct.unpack("!L", Sip)[0]

        except:
            continue

        try:
            Dip = socket.inet_aton(Dip)
            Dip = struct.unpack("!L", Dip)[0]

        except:
            continue

        if Sport=='': continue
        if Dport=='': continue

        try:

            if "Background" in label:
                label=0

            elif "Normal" in label:
                label = 0

            elif "Botnet" in label:
                label = 1
            
            if (proto in protoDict):
                x = protoDict[proto]
            else:
                x =x+1;
                protoDict[proto] = x;

            if flag==0:

                if label==0 and count1<5000:

                    xdata.append([float(dur), x, int(Sport), int(Dport), Sip, Dip, int(totB)])
                    ydata.append(label)
                    count1+=1

                elif label==1 and count2<5000:

                    xdata.append([float(dur), x, int(Sport), int(Dport), Sip, Dip, int(totB)])
                    ydata.append(label)
                    count2+=1

                elif count1>4999 and count2>4999:
                    
                    flag=1

            else:
                #Conjunto de dados de teste
                if label==0 and count3<5000:
                    xdataT.append([float(dur), x, int(Sport), int(Dport), Sip, Dip, int(totB)])
                    ydataT.append(label)
                    count3+=1
                elif label==1 and count4<5000:
                    xdataT.append([float(dur), x, int(Sport), int(Dport), Sip, Dip, int(totB)])
                    ydataT.append(label)
                    count4 += 1
                elif count3>4999 and count4>4999:
                    break

        except BaseException as e:
            print("Erro [+] "+str(e))
            continue

    #print(xdataT)
    #conjunto de dados para carregamento rápido 
    # Pickle - serialização para armazenamento em formato binário

    file = open('flowdata.pickle', 'wb')
    pickle.dump([np.array(xdata), np.array(ydata), np.array(xdataT), np.array(ydataT)], file)

    #print(ydataT)
    #dados = [np.array(xdata), np.array(ydata), np.array(xdataT), np.array(ydataT)]
    #retornar o conjunto de dados de treinamento e teste
    #print(dados)
    #print(type(dados))

    #retornar o conjunto de dados de treinamento e teste
    return np.array(xdata), np.array(ydata), np.array(xdataT), np.array(ydataT)

    #print(xdataT, ydataT)

if __name__ == "__main__":
    loaddata('./capture20110816-3.binetflow')

loaddata('./flowdata.csv')
