## git 使用&linux


### max + lgb 老的解决办法
    # brew install cmake
    # brew install libomp
    # brew install gcc@7 --without-multilib
    # pip3 install --no-binary :all: lightgbm


### 虚拟环境创建好，启动notebook, 可是发现无法import 安装好的包
    - sys.path 看下虚拟环境是否在系统环境路径里
    - 如果不在那就 sys.path.append('/Users/bruce/Desktop/qb_env_1/lib/python3.6/site-packages')
    - !pip list 看下是否安装过；
    - 可是我的问题是 ipython 默认的python 解释器不是我虚拟环境的解释器；我需要在虚拟环境里安装ipython/jupyter

    
### 

- git add : 把文件添加到暂存区

- git commit : 把文件提交到版本库


### 如果把错误方案提交到版本库（commited） 怎么办？

- git log 查询过去提交历史；

- git reset --mixed HEAD~1;  # 把版本库回滚到上一个版本；
	- mixed : 不改变工作区
	- hard: 工作区，暂存区， 版本库 全回滚；
	- soft: 只会滚版本库，不改变工作区，存区；


### 如果把错误方案add了，但是工作区已经修改正确，怎么办？
- 其中一个办法是 
	- git commit -m 'temp wrong commit'
	- git reset --mixed HEAD~1
	- git add ...
	
###  git push 后 停止在12% writing objects 怎么回事？
- 我用这个方法解决了
- git config --global http.postBuffer 524288000


### 另一个电脑上把本地代码与仓库同步
    - git remote -v # 查看远程仓库
    - git fetch origin master:temp # 在本地新建一个temp分支，并将远程origin仓库的master分支代码下载到本地temp分支
    - git merge temp # 把temp 合并到 本地master分支
    - git branch -d temp # 删除该分支

### git merge 出现矛盾怎么办？
    - git reset --hard # 如果发现矛盾可以把工作区和暂存区全部回滚到上一个版本，而后再merge就可以了
