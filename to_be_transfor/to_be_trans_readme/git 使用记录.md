## git 使用

### 

- git add : 把文件添加到暂存区

- git commit : 把文件提交到版本库


### 如果把错误方案提交到版本库（commited） 怎么办？

- git log 查询过去提交历史；

- git reset --mixed HEAD~1;  # 把版本库回滚到上一个版本；
	- mixed : 不改变工作区
	- hard: 工作区，暂存区， 版本库 全回滚；
	- soft: 只会滚版本库，不改变工作区，暂存区；

