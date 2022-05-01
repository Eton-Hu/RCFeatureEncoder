## Installation
Open command line with administrator mode, and type
```
C:\installation_path\RCFeatureEncoder>python setup.py install
```

## Import
```python
from RCFeatureEncoder import MeanEncoder
```

## Usage
You can copy following demo code for your first trail for mean encoder
```python
import pandas as pd
from RCFeatureEncoder import MeanEncoder
train_data = pd.read_csv("test_data/train.csv")
train_target = pd.read_csv("test_data/train_target.csv")
train_target = pd.Series(train_target.iloc[:,0])
test_data = pd.read_csv("test_data/test.csv")
encode_list = pd.read_csv("test_data/encodelist.csv")

print("debugging data:\n",train_data)
print("\ndebugging target:\n",train_data)
print("\nAll features:\n",train_data)
print("\nEncode list:\n",encode_list)

# Extract the columns needs to be encoded
train_data_to_encode = train_data.loc[:,list(encode_list.iloc[:,0])]
test_data_to_encode = test_data.loc[:,list(encode_list.iloc[:,0])]

# fill na values under each column
for column_i in train_data_to_encode.columns:
    train_data_to_encode.loc[:,column_i] = train_data_to_encode.loc[:,column_i].fillna(train_data_to_encode.loc[:,column_i].mode()[0])
    test_data_to_encode.loc[:,column_i] = test_data_to_encode.loc[:,column_i].fillna(train_data_to_encode.loc[:,column_i].mode()[0])

print("train_data_to_encode:\n",train_data_to_encode)
print("test_data_to_encode:\n",test_data_to_encode)

model = MeanEncoder()
trans_train = model.fit_transform(train_data_to_encode,train_target)
trans_test = model.transform(test_data_to_encode)

print("trans_train(with org):\n",trans_train)
print("trans_test(with org):\n",trans_test)

print("trans_train(without org):\n",trans_train.drop_org_features())
print("trans_test(without org):\n",trans_test.drop_org_features())
train_data = pd.read_csv("test_data/train.csv")
train_target = pd.read_csv("test_data/train_target.csv")
train_target = pd.Series(train_target.iloc[:,0])
test_data = pd.read_csv("test_data/test.csv")
encode_list = pd.read_csv("test_data/encodelist.csv")

print("debugging data:\n",train_data)
print("\ndebugging target:\n",train_data)
print("\nAll features:\n",train_data)
print("\nEncode list:\n",encode_list)

# Extract the columns needs to be encoded
train_data_to_encode = train_data.loc[:,list(encode_list.iloc[:,0])]
test_data_to_encode = test_data.loc[:,list(encode_list.iloc[:,0])]

# fill na values under each column
for column_i in train_data_to_encode.columns:
    train_data_to_encode.loc[:,column_i] = train_data_to_encode.loc[:,column_i].fillna(train_data_to_encode.loc[:,column_i].mode()[0])
    test_data_to_encode.loc[:,column_i] = test_data_to_encode.loc[:,column_i].fillna(train_data_to_encode.loc[:,column_i].mode()[0])

print("train_data_to_encode:\n",train_data_to_encode)
print("test_data_to_encode:\n",test_data_to_encode)

model = MeanEncoder()
trans_train = model.fit_transform(train_data_to_encode,train_target)
trans_test = model.transform(test_data_to_encode)

print("trans_train(with org):\n",trans_train)
print("trans_test(with org):\n",trans_test)

print("trans_train(without org):\n",trans_train.drop_org_features())
print("trans_test(without org):\n",trans_test.drop_org_features())
```
