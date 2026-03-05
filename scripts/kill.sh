ps aux|grep python|grep -v grep|cut -c 9-16|xargs kill -9
ps aux|grep python3|grep -v grep|cut -c 9-16|xargs kill -9