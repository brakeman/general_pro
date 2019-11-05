## 流程
    - 关键词： 测试集合 transform； 预处理，后处理； autoFeat;  业务逻辑特征
    
    
## 文件
    |--------       __init__.pu
    |--------       PrePostProcess.py
    |--------       AutoFeat.py
    |--------       TransForm.py
    |--------       HumanLook.py
    |--------       test
        |----       auto_combine_test.py
        |----       to_bins_test.py
        
## 除了本项目主要的通用特征衍生，还有业务逻辑特征
    - 1。 多个地址列，可能其中一列是 621785 这种身份证前6位地址，也可能是用户自己填的，可以衍生一个DIFF(dist1, dist2)用来判断用户是否虚报地址；
## finished
    - 分箱及其transform; 完成；
    - AutoFeat(transfrom空置率控制) 完成；
 
    
## 今天to be done;
    - lgb train with grid search best auc & return top k features  
    - null Class, 
	- 列
                - 相关性
		- 列缺
		- 列分布一致性	
	- 行 
    		- 异常点检测
    - 更多的transform 类型
## to be done:
    - 时序形表格特征提取/transform
    
    
    
    
    
