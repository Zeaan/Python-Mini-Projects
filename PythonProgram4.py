print("Sample")
import pandas as pd
import numpy as np
input=[["Amar",32,"Manager"],["Birla",25,"SalesPerson"],["Parul",23,"Trainee"]]
dfo=pd.DataFrame(input,columns=['Name','Age','Designation'],index=[1,2,3])
print(dfo)
