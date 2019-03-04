
class Utilities(object):

    @staticmethod
    def decode_gen(array):

        # index 24 - 39 floats
        rows = array.shape[0]
        cols = array.shape[1]

        list_of_lists = []

        for r in range(rows):
            details = []  # list
            for c in range(cols):
                #print(int_ndarray[r][c])
                if(c == 0):   # duration
                    details.append(int(array[r][c]))
                elif(c == 1):   # protocol type
                    if(int(array[r][c]) == 0):
                        details.append('tcp')
                    elif(int(array[r][c]) == 1):
                        details.append('udp')
                    elif(int(array[r][c]) == 2):
                        details.append('icmp')
                elif(c == 2):   # service
                    if(int(array[r][c]) == 0):
                        details.append('http')
                    elif(int(array[r][c]) == 1):
                        details.append('smtp')
                    elif(int(array[r][c]) == 2):
                        details.append('domain_u')
                    elif(int(array[r][c]) == 3):
                        details.append('auth')
                    elif(int(array[r][c]) == 4):
                        details.append('finger')
                    elif(int(array[r][c]) == 5):
                        details.append('telnet')
                    elif(int(array[r][c]) == 6):
                        details.append('eco_i')
                    elif(int(array[r][c]) == 7):
                        details.append('ftp')
                    elif(int(array[r][c]) == 8):
                        details.append('ntp_u')
                    elif(int(array[r][c]) == 9):
                        details.append('ecr_i')
                    elif(int(array[r][c]) == 10):
                        details.append('other')
                    elif(int(array[r][c]) == 11):
                        details.append('urp_i')
                    elif(int(array[r][c]) == 12):
                        details.append('private')
                    elif(int(array[r][c]) == 13):
                        details.append('pop_3')
                    elif(int(array[r][c]) == 14):
                        details.append('ftp_data')
                    elif(int(array[r][c]) == 15):
                        details.append('netstat')
                    elif(int(array[r][c]) == 16):
                        details.append('daytime')
                    elif(int(array[r][c]) == 17):
                        details.append('ssh')
                    elif(int(array[r][c]) == 18):
                        details.append('echo')
                    elif(int(array[r][c]) == 19):
                        details.append('time')
                    elif(int(array[r][c]) == 20):
                        details.append('name')
                    elif(int(array[r][c]) == 21):
                        details.append('whois')
                    elif(int(array[r][c]) == 22):
                        details.append('domain')
                    elif(int(array[r][c]) == 23):
                        details.append('mtp')
                    elif(int(array[r][c]) == 24):
                        details.append('gopher')
                    elif(int(array[r][c]) == 25):
                        details.append('remote_job')
                    elif(int(array[r][c]) == 26):
                        details.append('rje')
                    elif(int(array[r][c]) == 27):
                        details.append('ctf')
                    elif(int(array[r][c]) == 28):
                        details.append('supdup')
                    elif(int(array[r][c]) == 29):
                        details.append('link')
                    elif(int(array[r][c]) == 30):
                        details.append('systat')
                    elif(int(array[r][c]) == 31): 
                        details.append('discard')
                    elif(int(array[r][c]) == 32):
                        details.append('X11')
                    elif(int(array[r][c]) == 33):
                        details.append('shell')
                    elif(int(array[r][c]) == 34):
                        details.append('login')
                    elif(int(array[r][c]) == 35):
                        details.append('imap4')
                    elif(int(array[r][c]) == 36):
                        details.append('nntp')
                    elif(int(array[r][c]) == 37):
                        details.append('uucp')
                    elif(int(array[r][c]) == 38):
                        details.append('pm_dump')
                    elif(int(array[r][c]) == 39):
                        details.append('IRC')
                    elif(int(array[r][c]) == 40):
                        details.append('Z39_50')
                    elif(int(array[r][c]) == 41):
                        details.append('netbios_dgm')
                    elif(int(array[r][c]) == 42):
                        details.append('ldap')
                    elif(int(array[r][c]) == 43):
                        details.append('sunrpc')
                    elif(int(array[r][c]) == 44):
                        details.append('courier')
                    elif(int(array[r][c]) == 45):
                        details.append('exec')
                    elif(int(array[r][c]) == 46):
                        details.append('bgp')
                    elif(int(array[r][c]) == 47):
                        details.append('csnet_ns')
                    elif(int(array[r][c]) == 48):
                        details.append('http_443')
                    elif(int(array[r][c]) == 49):
                        details.append('klogin')
                    elif(int(array[r][c]) == 50):
                        details.append('printer')
                    elif(int(array[r][c]) == 51):
                        details.append('netbios_ssn')
                    elif(int(array[r][c]) == 52):
                        details.append('pop_2')
                    elif(int(array[r][c]) == 53):
                        details.append('nnsp')
                    elif(int(array[r][c]) == 54):
                        details.append('efs')
                    elif(int(array[r][c]) == 55):
                        details.append('hostnames')
                    elif(int(array[r][c]) == 56):
                        details.append('uucp_path')
                    elif(int(array[r][c]) == 57):
                        details.append('sql_net')
                    elif(int(array[r][c]) == 58):
                        details.append('vmnet')
                    elif(int(array[r][c]) == 59):
                        details.append('iso_tsap')
                    elif(int(array[r][c]) == 60):
                        details.append('netbios_ns')
                    elif(int(array[r][c]) == 61):
                        details.append('kshell')
                    elif(int(array[r][c]) == 62):
                        details.append('urh_i')
                    elif(int(array[r][c]) == 63):
                        details.append('http_2784')
                    elif(int(array[r][c]) == 64):
                        details.append('harvest')
                    elif(int(array[r][c]) == 65):
                        details.append('aol')
                    elif(int(array[r][c]) == 66):
                        details.append('tftp_u')
                    elif(int(array[r][c]) == 67):
                        details.append('http_8001')
                    elif(int(array[r][c]) == 68):
                        details.append('tim_i')
                    elif(int(array[r][c]) == 69):
                        details.append('red_i')
                elif(c == 3):   # flag
                    if(int(array[r][c]) == 0):
                        details.append('SF')
                    elif(int(array[r][c]) == 1):
                        details.append('S2')
                    elif(int(array[r][c]) == 2):
                        details.append('S1')
                    elif(int(array[r][c]) == 3):
                        details.append('S3')
                    elif(int(array[r][c]) == 4):
                        details.append('OTH')
                    elif(int(array[r][c]) == 5):
                        details.append('REJ')
                    elif(int(array[r][c]) == 6):
                        details.append('RSTO')
                    elif(int(array[r][c]) == 7):
                        details.append('S0')
                    elif(int(array[r][c]) == 8):
                        details.append('RSTR')
                    elif(int(array[r][c]) == 9):
                        details.append('RSTOS0')
                    elif(int(array[r][c]) == 10):
                        details.append('SH')
                elif(c == 4):
                    details.append(int(array[r][c]))
                elif(c == 5):
                    details.append(int(array[r][c]))
                elif(c == 6):
                    details.append(int(array[r][c]))
                elif(c == 7):
                    details.append(int(array[r][c]))
                elif(c == 8):
                    details.append(int(array[r][c]))
                elif(c == 9):
                    details.append(int(array[r][c]))
                elif(c == 10):
                    details.append(int(array[r][c]))
                elif(c == 11):
                    details.append(int(array[r][c]))
                elif(c == 12):
                    details.append(int(array[r][c]))
                elif(c == 13):
                    details.append(int(array[r][c]))
                elif(c == 14):
                    details.append(int(array[r][c]))
                elif(c == 15):
                    details.append(int(array[r][c]))
                elif(c == 16):
                    details.append(int(array[r][c]))
                elif(c == 17):
                    details.append(int(array[r][c]))
                elif(c == 18):
                    details.append(int(array[r][c]))
                elif(c == 19):
                    details.append(int(array[r][c]))
                elif(c == 20):
                    details.append(int(array[r][c]))
                elif(c == 21):
                    details.append(int(array[r][c]))
                elif(c == 22):
                    details.append(int(array[r][c]))
                elif(c == 23):
                    details.append(int(array[r][c]))
                elif(c == 24):  # floats
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 25):  
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 26):
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 27):
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 28):
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 29):
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 30):
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 31):
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 32):
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 33):
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 34):
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 35):
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 36):
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 37):
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 38):
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 39):
                    details.append(abs(float("%0.2f" % array[r][c])))
                elif(c == 40):
                    details.append(abs(float("%0.2f" % array[r][c])))

            list_of_lists.append(details)
        
        return list_of_lists

    @staticmethod
    def attacks_to_num(att):
        if(att == "normal"):
            return 1
        elif(att == "buffer_overflow"):
            return 2
        elif(att == "loadmodule"):
            return 3
        elif(att == "perl"):
            return 4
        elif(att == "neptune"):
            return 5
        elif(att == "smurf"):
            return 6
        elif(att == "guess_passwd"):
            return 7
        elif(att == "pod"):
            return 8
        elif(att == "teardrop"):
            return 9
        elif(att == "portsweep"):
            return 10
        elif(att == "ipsweep"):
            return 11
        elif(att == "land"):
            return 12
        elif(att == "ftp_write"):
            return 13
        elif(att == "back"):
            return 14
        elif(att == "imap"):
            return 15
        elif(att == "satan"):
            return 16
        elif(att == "phf"):
            return 17
        elif(att == "nmap"):
            return 18
        elif(att == "multihop"):
            return 19
        elif(att == "warezmaster"):
            return 20
        elif(att == "warezclient"):
            return 21
        elif(att == "spy"):
            return 22
        elif(att == "rootkit"):
            return 23

