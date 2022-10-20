作者想要实际测试一下深度学习的效果，在买带GPU的电脑还是用云服务之间果断选择后者。一个是目前只是想浅浅尝试一下，短期使用还是比买一台电脑划算；一个是念书时习惯了超算的便利，包括天河、曙光、学校以及各实习单位的大型机。AWS是全球最大的云服务器供应商，所以直接选择它了，当然也可以选用国内外其它的平台（可能会有更适合的平台，不在介绍范围内）。

本文详细记录了**“AWS注册-->配额申请-->实例创建-->固定IP绑定-->自动定时关闭设置-->登陆-->VScode远程连接”**的整个流程，供有需要的小伙伴参考。

## 1 注册AWS账户
这个过程参考clarmy的文章[《手把手教你在AWS上搭一个术语自己的博客网站》](https://mp.weixin.qq.com/s/Oa1W7Dv02i1X89SiDzecEA)。同时熟悉一下实例的概念，以及创建实例的过程。

## 2 创建AWS instance（实例）

### 2.1 p2实例的配额申请

考虑到深度学习很多时候需要GPU，推荐使用p2.xlarge实例，收费跟区域有关，这里我选择的是us-west-2区域，每小时0.98$（可以闲鱼上搜一下aws礼卡优惠券或者账号之类）。

为什么选us-west-2区域呢？因为不是每个区域都可以创建p2实例的，而且后续需要使用固定IP的时候，不是所有区域都支持这项服务，因此需要提前查好。

账户默认是没有p2实例的配额的，配额查询点[这里](https://docs.aws.amazon.com/zh_cn/general/latest/gr/aws_service_limits.html?id=docs_gateway)。

按照[教程](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html#vcpu-limits-request-increase)进行配额申请：

	1. 进入aws support的create case界面；
	2. 选择service limit increase, 填写表单，
	3. 表单内容示例如下：
  ![](https://files.mdnice.com/user/36794/7d7505a7-34d3-41e6-8e69-016747706a09.png)

我填写的case description如下图，可以参考。不过最后要再加一段，Since p2.xlarge is equipped with 4 vCPUs and 1 GPU，I may need all the resources approved, that is, increase the vCPU limit to 4. 不然像我的就是给开了p2但是vCPU的限额是1，还是无法开启4个CPU的实例。这个过程不太快，有邮件联系之后要及时回复询问进展。
![](https://files.mdnice.com/user/36794/f30ac862-48c2-4629-89de-eef5e0da0ffd.png)
### 2.2 创建p2实例

- 配额申请成功之后，选择EC2，创建实例，AMI选择时搜索deep learnig，选pytorch ubuntu的即可。这个镜像直接安装好了所需的一些基本软件，python、conda、cuda、pytorch等等。
![](https://files.mdnice.com/user/36794/111a5117-2226-42db-ba6e-6b67d36253cd.png)
![](https://files.mdnice.com/user/36794/3722733b-0fe7-48b9-a4ef-1e3729d2a112.png)
- 实例选择p2.xlarge；然后生成一个密钥，要注意保存好，后面登陆的时候要用。
![](https://files.mdnice.com/user/36794/7cf9cb3e-b4e1-492c-a6ba-143a898ba243.png)
- 网络设置这里放开80端口的防火墙，这样浏览器才能访问我们的网站，具体操作是点击下方的“add security group rule”，然后在新的配置框中把“端口范围”设为80。
![](https://files.mdnice.com/user/36794/bac9addc-c794-498c-a274-f12c886e13c3.png)
- 存储大小可以自己设置，有30G免费容量。
- 在右侧边的summary中确认一下信息，点lauch instance即可。
- 返回控制台，可以发现有一个实例，勾选之后，可以查看详情，其中公网IP就是我们登陆服务器时需要使用的IP地址。
![](https://files.mdnice.com/user/36794/b5799126-147a-4494-93bd-d77333c764d9.png)
但这个公网IP是动态的，每次关闭了重开就会变化，比较麻烦。所以下一小节介绍Elastic IP address的申请，并将其分配到这个实例上。
### 2.3 [固定IP地址关联](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/elastic-ip-addresses-eip.html#working-with-eips)
EC2窗口中选择Elastic IP，点右上角allocate elastic IP address，创建好之后选择分配好的地址，在右上角Actions里面选择Associate Elastic IP address，选择刚才创建的instance就好。
![](https://files.mdnice.com/user/36794/da018cc6-2235-4c7e-acdf-9f3155d21b56.png)
![](https://files.mdnice.com/user/36794/ff2f2338-44dc-4ba9-a69f-875351c9af03.png)
回到实例的界面，会看到公网地址变成这个固定的IP地址。
## 3 登陆服务器
选中实例，右击connect，找到合适的登陆方式。
![](https://files.mdnice.com/user/36794/0acad25f-086d-4d47-8832-5ce02df6586e.png)
![](https://files.mdnice.com/user/36794/296d1086-2791-45df-9c1a-c7e3157518a4.png)
其中ssh登陆的方法总是：找到密钥的位置，修改权限，输入`ssh -i test.pem ubuntu@你自己的公网IP）`即可。默认用户名是ubuntu。

不熟悉linux的朋友在登陆这块还是可以参考[clarmy的文章](https://mp.weixin.qq.com/s/Oa1W7Dv02i1X89SiDzecEA)。

## 4 设置自动定时关闭以免经费燃烧🔥
这一节介绍实例的每日定时关闭操作，如果每次都能记得关闭，不需要可以直接跳过第四节，记性不好如我，就需要这个功能。
### 4.1 [为 Lambda 函数创建 IAM policy 和执行角色](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_create-console.html#access_policies_create-json-editor)
- 搜索lambda，点击右上角的create policy，点进去之后选jason，把下面这段话复制粘贴进去，然后一直选next，review的时候起一个名字之后点击create policy即可。
```json
{ "Version": "2012-10-17", "Statement": [ { "Effect": "Allow", "Action": [ "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents" ], "Resource": "arn:aws:logs:*:*:*" }, { "Effect": "Allow", "Action": [ "ec2:Start*", "ec2:Stop*" ], "Resource": "*" } ] }
```
![](https://files.mdnice.com/user/36794/bbfb82e8-2afa-4d46-af48-4151bdd38515.png)
- 为 Lambda 创建 IAM 角色。
搜索roles，选择create roles， 在trusted entity type下选择aws service，use case下选择lambda，点击下一步
![](https://files.mdnice.com/user/36794/6cd605e1-aa64-4638-b768-c17edce220c0.png)
选择刚才创建的policy，点击下一步，role name自己取一个名称，我这里叫xiaowu_scheduler，点击create roles，就可以看到名为xiaowu_scheduler的角色了。
### 4.2 创建关闭 EC2 实例的 Lambda 函数
- 搜索lamba，点击右上角create function，选择author from scratch（从头开始创作），自己确定一个函数名称，如"StopEC2Instances"，Runtime选择python 3.9， permission下面展开更改原定设置执行角色，选择使用现有角色，选择刚刚创建的 IAM 角色。
![](https://files.mdnice.com/user/36794/5bb92ab1-0cf8-47fd-b291-ae5d693f25a0.png)
- 选择创建函数，在code的code source下粘贴如下函数，要把区域和实例id改成自己的，点击deploy。
```python
import boto3 
region = 'us-west-2' 
instances = ['i-12345cb6de4f78g9h', 'i-08ce9b2d7eccf6d26'] 
ec2 = boto3.client('ec2', region_name=region) 
def lambda_handler(event, context): 
  ec2.stop_instances(InstanceIds=instances) 
  print('stopped your instances: ' + str(instances))
```
![](https://files.mdnice.com/user/36794/4b2ffb9b-d187-43e4-8a6c-d7672f677aa4.png)
- 在lamda的function下选择刚创建的函数并点击test，可以看到log记录，也可以查看实例是否被关闭，从而判断是否执行成功。
![](https://files.mdnice.com/user/36794/e59f438e-d5b3-4fcb-af97-f827cf546400.png)
### 4.3 创建触发Lambda函数的EventBridge规则
- 搜索EventBridge，点击create rule，选择固定时间执行，我这里想要每天北京时20时进行关闭，换算成UTC时间要-8小时。
![](https://files.mdnice.com/user/36794/efefa96a-f0ea-453f-b032-bb3dad806e63.png)
之后一路选择next，target下选择aws service，select a target 选择刚才创建的lambda函数，create之后就大功告成，每天晚8点就会自动关闭实例了。
## 5 VSCode远程连接服务器
VSCode真是无敌好用的编辑器，也支持jupyter notebook和terminal的调试。
我这里网上搜了一个[vscode远程连接服务器写Jupyter Notebook的教程](https://www.jianshu.com/p/e8f377f498df)供大家参考。我这里也简述一下：在左边remote explorer点ssh那里左上角的加号，在弹出来的ssh command里面输入`ssh ubuntu@你的公网IP`, 右下角弹出的edit config file那里点确定，之后修改一下config file，把pem文件加进去，就可以正常连接啦。