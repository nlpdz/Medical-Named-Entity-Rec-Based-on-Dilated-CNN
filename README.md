## 基于膨胀神经网络（Dilated Convolutions）训练好的医疗领域的命名实体识别工具，这里主要给出模型源码，以及云部署方式供大家交流学习。
#### 环境
- 阿里云服务器：Ubuntu 16.04
- Python版本：3.6
- Tensorflow: 1.5

#### 第一步：来一个Flask实例，并跑起来：
使用的是Pycharm创建自带的Flask项目，xxx.py
```
from flask import Flask

app = Flask(__name__)

@app.route('/')

def hello_world():

return 'Hello World!'

if __name__ == '__main__':

app.run()
```

执行python xxx.py就可以运行
在浏览器中测试[http://127.0.0.1:5000，可以正常访问就ok。](http://127.0.0.1:5000```，可以正常访问就ok。)
若直接在dos窗口中：输入命令[ wget http://127.0.0.1:5000 ]() 也可测试。
#### 第二部：服务器配置
1. 服务器python版本为3.x
2. 安装pip （sudo apt-get install python-pip）
3. 安装Nginx，[https://lnmp.org/](https://lnmp.org/%60%60%60)  或者 （sudo apt-get install nginx）
#### 第三步：安装Gunicorn
1. 安装虚拟环境
```
pip3 install virtualenv
```

新建一个目录用作网站根目录，这里使用lnmp的根路径/home/wwwroot/myflask，并进入该目录
```
cd/home/wwwroot/myflask
```

创建一个独立的Python环境，命名为``envFflask```，完成后激活该环境
```
virtualenv envFlask--python=python3.6

source envFlask/bin/activate
```

2. 安装gunicorn和flask
在虚拟环境下分别执行
```
pip3 install gunicorn

pip3 install flask
```

同时将项目文件xxx.py，上传到/home/wwwroot/myflask，尝试执行python xxx.py
在本机浏览器访问 [http://ip:8000，ip是服务器ip，如果正常的话，环境就没问题了。](http://ip:5000```，ip是服务器ip，如果正常的话，环境就没问题了。)
但是这个时候还是使用的python自带的web服务器，下面我们使用gunicorn
#### 第四步：使用gunicorn
执行命令gunicorn -w 3 -b 127.0.0.1:8000 xxx:app
在本机浏览器访问 [http://ip:5000，ip是服务器ip，如果正常的话，环境就没问题了。](http://ip:5000```，ip是服务器ip，如果正常的话，环境就没问题了。)
解释命令：-w 3表示开3个线程，-b 120.0.0.1:8000表示路径设置，xxx:app：xxx表示入口文件，app表示主函数
疑问：为什么python自带的就可以运行了， 还需要这么复杂？
#### 第五步：配置Nginx
先按Ctrl+C，停止gunicorn，
Nginx配置文件地址：/etc/nginx/sites-enabled/default
如下修改：
```
server

{

listen 9004;   # 自己设置，同时在阿里云防火墙打开该端口

server_name 39.108.91.172; # 这是HOST机器的外部域名，用地址也行

location / {

proxy_pass http://127.0.0.1:8000; # 这里是指向 gunicorn host 的服务地址

proxy_set_header Host $host;

proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

}
```

重启Nginx：
```
sudo service nginx restart
```

此时，就可以使用ip还是无法正常访问，因为我们前面停止了gunicorn，
现在使用gunicorn -w 3 -b 127.0.0.1:8000 xxx:app启动起来，
浏览器测试正常，下一步Go--
#### 第六步，将gunicorn注册为系统服务，后台运行
我是从这里学习的[https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-16-04](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-16-04%60%60%60)

创建一个systemd单元文件
我们需要处理的下一件事是systemd服务单元文件。创建一个systemd单元文件将允许Ubuntu的init系统自动启动Gunicorn，并在服务器启动时为Flask应用程序提供服务。
在/ etc / systemd / system目录中创建以.service结尾的单元文件(myflask.service)以开始：
给出我的配置文件myflask.service
```
[Unit]

Description=Gunicorn instance to serve the falcon application

After=network.target

[Service]

User=root

Group=www-data

Environment="PATH=/home/wwwroot/myflask/envFlask/bin"

WorkingDirectory=/home/wwwroot/myflask

ExecStart=/home/wwwroot/myflask/envFlask/bin/gunicorn -w 3 -b 127.0.0.1:8000 xxx:app

ExecReload=/bin/kill -s HUP $MAINPID

ExecStop=/bin/kill -s TERM $MAINPID

[Install]

WantedBy=multi-user.target
```


在/ etc / systemd / system目录输入以下命令：
```
sudo systemctl start myflask 
```
//启动gunicorn后台管理


重新在虚拟环境的项目目录启动gunicorn命令，开始后台运行gunicorn；：
```
gunicorn -w 3 -b 127.0.0.1:8000 xxx:app
```

关闭虚拟环境：
```
deactivate
```

这里可以使用ps -ef查看进程运行情况.
#### 第七步：更新替代
上传替代文件（我们的模型，代码文件），在/ etc / systemd / system目录重启service，再运行gunicorn
```
sudo systemctl start myflask //启动

sudo systemctl stop myflask //停止
```
#### 第八步：修改主文件 main.py
- main.py 重命名为 xxx.py
- 将文件中所有的  os.path.getcwd()  代码更改为 ：os.path.dirname(os.path.abspath( __file__ ))
- app.run(host='127.0.0.1',port=5002)  更改为   app.run()
- 屏蔽了断言(应该没必要)：
```
# assert FLAGS.clip < 5.1, "gradient clip should't be too much"
# assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
# assert FLAGS.lr > 0, "learning rate must larger than zero"
# assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]
```
#### 第九步：更改gunicorn的启动命令
- gunicorn -w 3 -b 127.0.0.1:8000 xxx:app
该命令中  -w ,-b ,作为参数出入了被调用的xxx.py文件，而xxx.py文件中没有接受该参数的flages
导致出错！（Tensorflow中flages的特别之处）
- 解决方法一：增加  flags.DEFINE_string/_int接受 -w,3,-b，127.0.0.1:8000  参数， 后面可以仿照这改语法使用参数。  如：
```
flags.DEFINE_int("w", 3, "Set the number of threads")
```
- 解决方法二：使用gunicorn启动项目时，不要传参了！ gunicorn xxx:app  即可！！
然后，需要更改/et /systemd/system 中的myflask.service配置参数。也将 -w,-b ,127.0.0.1:8000 去掉！ 

#### 第十步：将页面Unicode编码的结果改为中文
- 更改xxx.py中的代码：
```
import json
...
...
# return jsonify(aa)
return json.dumps(aa, ensure_ascii=False)
```
- 在/et/systemd/system目录下重启的myflask.service服务，重启gunicorn, 完成！

- 测试地址：http://39.108.91.172:9004/?inputStr=%22%E4%B9%99%E8%82%9D%E5%92%8C%E5%86%A0%E5%BF%83%E7%97%85%E9%82%A3%E4%B8%AA%E4%B8%A5%E9%87%8D%22

- 可利用爬虫技术反复调用该接口
