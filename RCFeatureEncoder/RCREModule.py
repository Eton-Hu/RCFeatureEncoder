import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .RCEncodedForm import *

# Mean encoder class
class MeanEncoder:
    def __init__(
                    self,
                    n_split=5,
                    target_type = 'classification',
                    weight_func = lambda x:1/(1+np.exp((x-2)/1)),
                    delete_org = False
                ):
        #Checking split K
        if n_split >= 1:
            self.n_split = n_split
        else:
            self.n_split = 1
        
        #Checking target type
        if target_type != 'classification' and target_type != 'regression':
            raise ValueError("Such task is not supported:",target_type)
        self.target_type = target_type

        #Checking weight function
        try:
            weight_func_result_high = weight_func(0)
            weight_func_result_low = weight_func(100)
        except BaseException as _:
            raise ValueError("Invalid weight function:",weight_func)
        finally:
            if weight_func_result_low<0 or weight_func_result_low > 1:
                raise ValueError("Invalid weight function:",weight_func)
            if weight_func_result_high<0 or weight_func_result_high > 1:
                raise ValueError("Invalid weight function:",weight_func)
            if weight_func_result_high<weight_func_result_low:
                raise ValueError("Invalid weight function:",weight_func)
        self.weight_func = weight_func

        #check bool params
        if not isinstance(delete_org, bool):
            raise ValueError("\"delete_org\" should be a boolen type parameter",delete_org)
        self.delet_org = delete_org

        self.data_len = None
        self.return_data = None
        self.feature_names = None
        self.maps = dict()

    # for print calling
    def __str__(self):
        if self.return_data is None:
            return "A Non-trained MeanEncoder module, pleas train the module first and then print generated data"
        else:
            return str(self.return_data)
    
    def __iter__(self):
        self.it = iter(self.return_data)
        return self.it
    
    def __next__(self):
        next(self.it)

    def fit_classification(self):
        # Build target_class list
        self.target_class = self.target_colum.drop_duplicates()
        self.target_class.index = np.arange(0,len(self.target_class))
        self.target_class_num = len(self.target_class.index)

        # Build feature_values list
        self.feature_unique = self.feature_values.drop_duplicates()
        self.feature_unique.index = np.arange(0,len(self.feature_unique))
        self.feature_values_num = len(self.feature_unique)

        # Build empty [Feature Values -> Expanded Feature Values] mapping dict
        self.feature_mapping = pd.DataFrame(
                                                data = np.zeros((self.feature_values_num+1 ,self.target_class_num)),
                                                index = list(self.feature_unique)+["ME_default"],
                                                columns = self.target_class
                                            )
        
        if self.n_split == 1:
            train_index_list = [self.feature_values.index]
        else:
            skf = StratifiedKFold(n_splits = self.n_split, shuffle = False)
            train_index_list = [x for x,_ in skf.split(self.feature_values, self.target_colum)]
        # split data into k-fold, and merge the result by averaging them
        for train_index in train_index_list:
            # Split data
            train_data,train_target = self.feature_values.iloc[train_index],self.target_colum.iloc[train_index]
            # Length of all data
            datalen = len(train_index)

            # Target valute iterater, Calculate each target value one by one
            for target_class_i in self.target_class:
                # New Feature Values = (1-a)*P(target_class_i|feature_value_i)+a*P(target_class_i)
                P_target_class_i = len( train_target[train_target==target_class_i] )/datalen

                # Feature iterator
                for feature_value_i in list(self.feature_unique)+["ME_default"]:
                    # All data under this feature value
                    dtfv = train_data[train_data==feature_value_i]

                    #Target values under this FeatureValue
                    tvtf = train_target[(train_target==target_class_i) & (train_data==feature_value_i)]
                    if len(dtfv) == 0:
                        P_target_class_i_FeatureValue = 0
                    else:
                        P_target_class_i_FeatureValue = len(tvtf)/len(dtfv)
                    factor_a = self.weight_func(len(dtfv))

                    # Fault value handling
                    if factor_a > 1 or factor_a < 0:
                        raise ValueError("Invalid weight function:",self.weight_func)

                    # accumulating the calculation result of this iteration
                    temp_calculated_result = (1-factor_a)*P_target_class_i_FeatureValue + factor_a * P_target_class_i

                    if temp_calculated_result < 0.01:
                        temp_calculated_result = 0

                    # Update feature_mapping
                    self.feature_mapping.loc[feature_value_i,target_class_i] += temp_calculated_result/self.n_split
                pass    #Feature iterator
            pass    #Target valute iterater
        pass    #k-fold iterator

    def fit_regression(self):
        # Build feature_values list
        self.feature_unique = self.feature_values.drop_duplicates()
        self.feature_unique.index = np.arange(0,len(self.feature_unique))
        self.feature_values_num = len(self.feature_unique)

        # Build empty [Feature Values -> Expanded Feature Values] mapping dict
        self.feature_mapping = pd.DataFrame(
                                                data = np.zeros((self.feature_values_num+1 ,1)),
                                                index = list(self.feature_unique)+["ME_default"],
                                                columns = ["Transmitted from " + self.feature_values.name]
                                            )
                                            
        if self.n_split == 1:
            train_index_list = [self.feature_values.index]
        else:
            skf = StratifiedKFold(n_splits = self.n_split, shuffle = False)
            train_index_list = [x for x,_ in skf.split(self.feature_values, self.target_colum)]
        # split data into k-fold, and merge the result by averaging them
        for train_index in train_index_list:
            # Split data
            train_data,train_target = self.feature_values.iloc[train_index],self.target_colum.iloc[train_index]

            # Mean value of all target values
            all_tar_m = train_target.mean()

            # Feature iterator
            for feature_value_i in list(self.feature_unique)+["ME_default"]:
                # New Feature Values = (1-a)*this_tar_m + a*all_tar_m

                # Target values of this FeatureValue
                tvtf = train_target[train_data==feature_value_i]
                # Mean value of target values under this feature value
                if len(tvtf) > 0:
                    this_tar_m = tvtf.mean()
                else:
                    this_tar_m = 0

                factor_a = self.weight_func(len(tvtf))

                # Fault value handling
                if factor_a > 1 or factor_a < 0:
                    raise ValueError("Invalid weight function:",self.weight_func)

                # accumulating the calculation result of this iteration
                temp_calculated_result = (1-factor_a) * this_tar_m + factor_a * all_tar_m

                if temp_calculated_result < 0.01:
                    temp_calculated_result = 0

                # Update feature_mapping
                self.feature_mapping.loc[feature_value_i,"Transmitted from " + self.feature_values.name] += temp_calculated_result/self.n_split
            pass    #Feature iterator
        pass    #k-fold iterator

    # fit data from one feature column
    def fit_from_one_col(self, feature_column, target_colum):
        try:
            # Check validation via pd.Series
            self.target_colum = pd.Series(target_colum)
            self.feature_values = pd.Series(feature_column)
        except BaseException as _:
            raise ValueError("feature_column or target_colum should be a 1D iteratable veriable")
        
        # Check target size and feature data size
        if len(self.target_colum) != len(self.feature_values):
            self.target_colum = None
            self.feature_values = None
            raise ValueError("Size of target data should be the same as Size of feature data")
        
        # Init or check datalen
        if self.data_len is None:
            self.data_len = len(self.target_colum)
        else:
            if self.data_len != self.data_len:
                raise ValueError("Size of target data and feature data should be as the same as previous data")
        
        # NaN data was not supported yet
        if self.target_colum.hasnans or self.feature_values.hasnans:
            raise ValueError("NaN handling is not supported yet")

        # Init data in return
        if self.return_data is None:
            self.return_data = encoded_form(pd.DataFrame(self.feature_values), self.target_colum, [self.feature_values.name])
        else:
            self.return_data.insert_columns(pd.DataFrame(self.feature_values))
            pass
        
        # Extract all classes of target
        if self.target_type == 'classification':
            self.fit_classification()
        else:
            self.fit_regression()

        # mapping the data to return
        # Build temp array to store the data
        tl = np.zeros((len(self.feature_values),len(self.feature_mapping.columns)))
        for feature_value_idx in range(len(self.feature_values)):
            for target_idx in range(len(self.feature_mapping.columns)):
                tl[feature_value_idx,target_idx] = self.feature_mapping.loc[
                                                                                self.feature_values.iloc[feature_value_idx], 
                                                                                self.feature_mapping.columns[target_idx]
                                                                            ]
        temp_column= [str(self.feature_values.name)+"@ " + str(x) for x in self.feature_mapping.columns]
        # append mapped data to return
        self.return_data.add_columns(tl,temp_column)
        del tl
        del temp_column
        
        return self.return_data

    def fit(self,features, targets):
        try:
            # Check validation via pd.Series
            self.features = pd.DataFrame(features)
            self.targets = pd.Series(targets)
        except BaseException as _:
            raise ValueError("features or targets should be iteratable veriables \n\
                            features is a 2D array like or dict like variable \n\
                            targets is a 1D array like or list like variable")
        
        # Check target size and feature data size
        if len(self.features) != len(targets):
            self.target_colum = None
            self.feature_values = None
            raise ValueError("Size of target data should be the same as Size of feature data")
        
        # Creat feature name
        self.feature_names = self.features.columns
        
        # Train the module one column by one column
        for feature_column_i in self.features.columns:
            self.fit_from_one_col(self.features.loc[:,feature_column_i], self.targets)
            self.maps[feature_column_i] = self.feature_mapping.copy()

    # transform after fitting
    def fit_transform(self,features, targets):
        self.fit(features, targets)
        if self.delet_org == True:
            self.return_data.drop_org_features(inplace=True)
        return self.return_data

    # transform a data set based on the trained module
    def transform(self,features):
        # check validation of the input features
        try:
            features = pd.DataFrame(features)
        except BaseException as _:
            raise ValueError("feature should be an iteratable veriable")
        
        # Check columns size
        temp_new_col = list(features.columns)
        temp_old_col = list(self.features.columns)
        temp_new_col.sort()
        temp_old_col.sort()
        if temp_new_col != temp_old_col:
            del temp_new_col
            del temp_old_col
            del features
            raise ValueError("Input feature columns which is:\n{}\n should have the same contents with \
                            the module columns:\n{}".format(features.columns,self.features.columns))
        del temp_new_col
        del temp_old_col

        # NaN handling is not in the scope
        if features.isnull().values.any():
            raise ValueError("NaN handling is not supported yet")
        
        # reset return data
        self.return_data.clear_data()
        self.return_data.__init__(features,np.full(len(features), np.nan),columns=features.columns)

        # start mapping
        for column_i in list(features.columns):
            self.feature_mapping = self.maps[column_i]
            # Build temp array to store the data 
            tl = np.zeros((len(features),len(self.feature_mapping.columns)))
            for feature_value_idx in range(len(features)):
                for target_idx in range(len(self.feature_mapping.columns)):
                    if features.loc[:,column_i].iloc[feature_value_idx] in self.feature_mapping.index:
                        tl[feature_value_idx,target_idx] = self.feature_mapping.loc[
                                                                                        features.loc[:,column_i].iloc[feature_value_idx], 
                                                                                        self.feature_mapping.columns[target_idx]
                                                                                    ]
                    else:
                        tl[feature_value_idx,target_idx] = self.feature_mapping.loc[
                                                                                        "ME_default", 
                                                                                        self.feature_mapping.columns[target_idx]
                                                                                    ]
            temp_column= [str(column_i)+"@ " + str(x) for x in self.feature_mapping.columns]
            # append mapped data to return
            self.return_data.add_columns(tl,temp_column)
            del tl
            del temp_column

        if self.delet_org == True:
            self.return_data.drop_org_features(inplace=True)
        return self.return_data

# Debugging
if __name__=='__main__':
    # build test set
    train_data = pd.read_csv("../test_data/train.csv")
    train_target = pd.read_csv("../test_data/train_target.csv")
    train_target = pd.Series(train_target.iloc[:,0])
    test_data = pd.read_csv("../test_data/test.csv")
    encode_list = pd.read_csv("../test_data/encodelist.csv")

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