import socket   #for sockets
import sys  #for exit
 

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("allo")
host = "192.168.43.189"
port = 8080
 

s.connect((host , port))
print ('Socket Connected to ' + host )

 
#Now receive data

reply = s.recv(4096)
print(reply)
print("asdasd")