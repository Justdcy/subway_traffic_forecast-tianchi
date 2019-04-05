# subway_traffic_forecast-tianchi
萌新开源，大佬些多给点指导。
天池全球城市计算AI挑战赛-地铁人流量预测，
A榜22/2319，该代码是A榜代码，如果能有所收获，老铁右上角，star一下，感谢！
队友：buger，taoberica、selina雪，感谢鱼佬baseline，
A榜代码有部分是借鉴鱼佬开源代码，
没能进入决赛也就不开源淘汰赛的代码了。
数据集下载：

链接: https://pan.baidu.com/s/1iLHomv5NRodB_3jr7FcFow 提取码: arse 

比赛链接;https://tianchi.aliyun.com/competition/entrance/231708/introduction?spm=5176.12281957.1004.5.38b04c2alLBS7L

目前还有一些未来得及验证的想法，有兴趣的大佬些可以试试看。

a.将间隔十分钟改为间隔五分钟，相对增加了数据量

b.将shfit后的前三天删掉，因为shift后前三天引入了很多0

c.除了shift最近三天的策略，还可以试试shift最近两天+上一周相对应的week的数据

d.最开始也试了lgb模型，效果比xgb差点，可以将xgb和lgb采用blending融合

如有疑问，欢迎学习交流，QQ:1796320597
