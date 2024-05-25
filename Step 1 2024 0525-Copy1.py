#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import torch
import torch.nn.functional as F

#递推求下一个T的函数
def GW2(X, Y, T_t, p, q, epsilon=0.1, max_iter=100):
    C_t = -2 * torch.matmul(torch.matmul(X,T_t),Y.t())
    K_t = torch.exp(-C_t / epsilon)
    v = torch.ones_like(q)
    u = torch.ones_like(p)
    for _ in range(max_iter):
        u = p / torch.matmul(K_t, v)
        v = q / torch.matmul(K_t.t(), u)
        U = torch.diag(u.squeeze())
        V = torch.diag(v.squeeze())
        T = torch.matmul(torch.matmul(U,K_t),V)
    return T

if __name__ == '__main__':
    device='cpu'
    X = torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=torch.float32)
    Y = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)

    p = torch.tensor([0.5, 0.2, 0.3], dtype=torch.float32).unsqueeze(1) # 令p和q为PyTorch张量
    q = torch.tensor([0.4, 0.6], dtype=torch.float32).unsqueeze(1)
    N = 5  #大循环T的迭代次数 
    T_0 = torch.matmul(p,q.t()) 

    C_0 = -2 * torch.matmul(torch.matmul(X,T_0), Y.t())  
    
    v_0 = torch.ones_like(X, device=device) / X.shape[1]  #初始化v0

    T_list = []   #验证和python ot包的结果是否一样
    T = GW2(X, Y, T_0, p, q)
    T_list.append(T)  
    for i in range(N):
        T = GW2(X, Y, T, p, q)
        T_list.append(T)
        
    print(T_list)


# In[ ]:


import numpy as np
import torch
import torch.nn.functional as F

#递推求下一个T的函数
def GW2(X, Y, T_t, p, q, epsilon=0.1, max_iter=100):
    C_t = -2 * torch.matmul(torch.matmul(X,T_t),Y.t())
    K_t = torch.exp(-C_t / epsilon)
    v = torch.ones_like(q)
    u = torch.ones_like(p)
    for _ in range(max_iter):
        u = p / torch.matmul(K_t, v)
        v = q / torch.matmul(K_t.t(), u)
        U = torch.diag(u.squeeze())
        V = torch.diag(v.squeeze())
        T = torch.matmul(torch.matmul(U,K_t),V)
    return T

if __name__ == '__main__':
    device='cpu'
    X = torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=torch.float32)
    Y = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)

    p = torch.tensor([0.5, 0.2, 0.3], dtype=torch.float32).unsqueeze(1) # 令p和q为PyTorch张量
    q = torch.tensor([0.4, 0.6], dtype=torch.float32).unsqueeze(1)
    N = 1000 
    T_0 = torch.matmul(p,q.t()) 

    C_0 = -2 * torch.matmul(torch.matmul(X,T_0), Y.t())  #初始化C0
    
    v_0 = torch.ones_like(X, device=device) / X.shape[1]  #初始化v0，X.shape[2]张量X第三个维度的长度

    # T_list = []   #验证和python ot包的结果是否一样
    T = GW2(X, Y, T_0, p, q)
    # T_list.append(T)  
    for i in range(N):
        T = GW2(X, Y, T, p, q)
        # T_list.append(T)
        
    print(T)




# In[ ]:




