import time
import random
class user():
    def __init__(self,id,asks):
        self.id = id
        self.asks = asks
    def __str__(self):
        return "用户编号:{0},用户请求数目:{1}".format(self.id,len(self.asks))
class ask():
    def __init__(self,ask_id,ask_num,ask_time,time_bar):
        self.ask_id = ask_id
        self.ask_num=ask_num
        self.ask_time=ask_time
        self.time_bar = time_bar
    def __str__(self):
        return "请求编号:{0},请求时间:{1},请求时间限制:{2}".format(self.ask_id,self.ask_time,self.time_bar)
class handle():
    def __init__(self,users,all_ask_id):
        self.ask_ids = all_ask_id
        self.users = users
        self.ask_dict = {}  #所有请求的数目累计字典
        self.ask_time = {}  #所有的请求的最早时间累计字典
        self.ask_time_bar = {} #所有的请求对的最短时间限制
        time_zero = time.time()+10000000
        self.time_start = time.time()
        self.ask_sin_score = {}
        for i in self.ask_ids:
            self.ask_dict.update({i:0})
            self.ask_time.update({i:time_zero})
            self.ask_time_bar.update({i:100000000})
            self.ask_sin_score.update({i:-1})
        for u in self.users:
            for a in u.asks:
                self.ask_dict[a.ask_id]+=1
                if(a.ask_time<self.ask_time[a.ask_id]):
                    self.ask_time[a.ask_id] = a.ask_time
                time_has_bar = a.time_bar-self.time_start  #距离用户请求终止还剩下的时间
                if(time_has_bar<self.ask_time_bar[a.ask_id]):
                    self.ask_time_bar[a.ask_id]=time_has_bar
    def rxw(self):
        print("RXW算法处理结果如下 ：")
        ask_tuple = sorted(self.ask_dict.items(), key=lambda x: x[1], reverse=True)
        is_prined = [0]*len(ask_tuple)
        self.ask_dict = dict(ask_tuple)
        # print(ask_tuple)
        # self.ask_user = dict(sorted(self.ask_user.items(), key=lambda x: x[1], reverse=True))
        def core(i,time):
            if(i+1<len(ask_tuple) and ask_tuple[i][1]==ask_tuple[i+1][1]):
                if(time<self.ask_time[ask_tuple[i+1][0]]):
                    if(is_prined[i]==0) : print("反馈请求编号为{0}的请求".format(ask_tuple[i][0]))
                    is_prined[i]=1
                else:
                    core(i+1,self.ask_time[ask_tuple[i+1][0]])
            if(is_prined[i]==0) : print("反馈请求编号为{0}的请求".format(ask_tuple[i][0]))
            is_prined[i] = 1
        for i in range(len(ask_tuple)):
            core(i,self.ask_time[ask_tuple[i][0]])
    def sin(self):
        print("按照SIN算法处理结果如下：")
        for id in self.ask_ids:
            self.ask_sin_score[id]=float(self.ask_time_bar[id]/self.ask_dict[id])
        sin_score_tuple = sorted(self.ask_sin_score.items(),key = lambda x:x[1],reverse =False)
        for id,num in sin_score_tuple:
            print("反馈请求编号为{0}的问题".format(id))
if __name__ == "__main__":
    num_users = 5  #有5个用户
    ask_num_every_one = 7  #每个用户有7个请求
    user_id=0
    all_users=[]
    all_ask_id = set()
    for i in range(num_users):
        user_asks = []  #用户列表
        user_this = user(user_id,[])
        user_id+=1
        #生成这个用户的请求列表
        for j in range(ask_num_every_one):
            ask_time = time.time() #这个请求所提交的时间
            ask_bar = random.randint(100,200)+ask_time #随机生成这个请求的时间限制,100秒到200秒之后系统需要反馈
            ask_id = random.randint(3,7) #随机生成这个请求的编号，3号到7号之间的随机数
            all_ask_id.add(ask_id) #将请求的编号加入到问题编号集合中
            ask_obj = ask(ask_id,ask_num_every_one,time.time(),ask_bar) #生成请求
            time.sleep(0.3)   #程序暂停0.3秒
            user_this.asks.append(ask_obj) #将请求添加到user的请求集合中
        all_users.append(user_this)
    for u in all_users:
        print(u)
        print("#########################该用户的请求情况如下############################")
        for ask in u.asks:
            print(ask)
        print()

    #开始处理用户的请求
    handle_obj = handle(all_users,all_ask_id)
    #RXW算法
    handle_obj.rxw()
    #SIN算法
    handle_obj.sin()

