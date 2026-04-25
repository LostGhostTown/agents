from agent.mylm import qianFan, chain
#对话模型
resp=chain.invoke("1+1等于几")
print(resp)
#print(resp.content)

#for chunk in qianFan.stream("你好"):
#    print(chunk)
#    print(type(chunk))       