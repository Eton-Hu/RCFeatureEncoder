import numpy as np
import pandas as pd

class encoded_form(object):
    # init the encoded data
    def __init__(
                    self, 
                    initdata = None, 
                    inittarget = None, 
                    columns = None
                ):
        if initdata is not None:
            self.data = np.array(initdata)
            if self.data.ndim != 2:
                raise ValueError("feature values should be in 2 dimension form, instead of {}".format(self.data.ndim))
            self.raw_size = self.data.shape[0]
            self.column_size = self.data.shape[1]
        else:
            self.data = None
            self.raw_size = 0
            self.column_size = 0
        
        if inittarget is not None:
            self.target = np.array(inittarget)
            if self.target.ndim != 1:
                self.target = None
                self.data = None
                self.raw_size = 0
                self.column_size = 0
                raise ValueError("Target values should be in 1 dimension form",inittarget)
        else:
            self.target = None

        self.data_type = type(initdata)
        pddf_type = type(pd.DataFrame())
        # See if there are columns infomations in the orginal data
        if columns is None:
            # there are columns in pd.dataframe type data
            if isinstance (initdata,pddf_type):
                self.columns = list(initdata.columns)
            else:
                # Auto generate columns if it was not defined
                self.columns = range(self.column_size)
        else:
            # if the columns are manually inputted then check the legitimacy
            if len(list(columns)) == self.column_size:
                self.columns = list(columns)
            else:
                raise ValueError("Invalid columns input")
        self.feature_column_size = self.column_size

    # for print calling
    def __str__(self):
        return str(pd.DataFrame(self.data,columns=self.columns))
    
    def __del__(self):
        self.clear()
        del self

    def __iter__(self):
        self.it = iter(pd.DataFrame(self.data,columns=self.columns))
        return self.it
    
    def __next__(self):
        next(self.it)

    # add some columns to the data
    def add_columns(self, input_data, columns=None):
        data = np.array(input_data)
        if data.shape[0] != self.raw_size:
            raise ValueError("new data raw should equivilent with \"encoded_form.raw\":",data)
        if data.ndim != self.data.ndim:
            raise ValueError("data dimension should be less than 2: ",data)

        pddf_type = type(pd.DataFrame())
        # See if there are columns infomations in the orginal data
        if columns is None:
            # there are columns in pd.dataframe type data
            if isinstance (input_data,pddf_type):
                self.columns.extend(list(input_data.columns))
            else:
                # Auto generate columns if it was not defined
                self.columns.extend(range(self.column_size,self.column_size+data.shape[1]))
        else:
            # if the columns are manually inputted then check the legitimacy
            if len(list(columns)) == data.shape[1]:
                self.columns.extend(list(columns))
            else:
                raise ValueError("Invalid columns input")

        self.column_size += data.shape[1]
        self.data = np.hstack((self.data,data))
        

    # insert some columns to the data
    def insert_columns(self, input_data ,columns=None):
        data = np.array(input_data)
        if data.shape[0] != self.raw_size:
            raise ValueError("new data raw should equivilent with \"encoded_form.raw\":",data)
        if data.ndim != self.data.ndim:
            raise ValueError("data dimension should be 2, instead of: {}".format(data.ndim))

        insert_from = self.feature_column_size

        # A tool for inserting multipul columns into an array
        def list_insert(main_list,sub_list,index):
            idx = index
            for i in sub_list:
                main_list.insert(idx,i)
                idx += 1
            return main_list

        pddf_type = type(pd.DataFrame())
        # See if there are columns infomations in the orginal data
        if columns is None:
            # there are columns in pd.dataframe type data
            if isinstance (input_data,pddf_type):
                self.columns = list_insert(self.columns,list(input_data.columns),insert_from)
            else:
                # Auto generate columns if it was not defined
                self.columns.extend(range(self.column_size, self.column_size+data.shape[1]))
        else:
            # if the columns are manually inputted then check the legitimacy
            if len(list(columns)) == data.shape[1]:
                self.columns = list_insert(self.columns,list(columns),insert_from)
            else:
                raise ValueError("Invalid columns input")

        #Update column number
        self.feature_column_size += data.shape[1]
        self.column_size += data.shape[1]

        # Insert data to np column
        self.data = np.insert(self.data,range(insert_from,insert_from+data.shape[1]),values=data,axis=1)


    # delete 1 column to the data
    def delete_column(self, column_index):
        if isinstance (column_index,'int'):
            np.delete(self.data, column_index, axis = 1)
            del self.columns[column_index]
        else:
            raise ValueError("column_index should be an interger",column_index)
        self.column_size -= 1

        # Indicate it is deleting an original feature column
        if column_index < self.feature_column_size:
            self.feature_column_size -= 1
    
    # delete columns range from data
    def delete_mul_columns(self, column_index):
        # i.e. if column_index = [1,3], then delete colomn 1,2,3, like np.delete()
        column_index = list(column_index)
        if len(column_index) != 2:
            raise ValueError("column_index should have 2 elements",column_index)
        np.delete(self.data, column_index, axis = 1)
        del self.columns[column_index[0]:column_index[1]]
        self.column_size -= column_index[1] - column_index[0]

        # Indicate it is deleting some original feature column
        if column_index[0] < self.feature_column_size:
            deleted_column_num = self.feature_column_size - column_index[0]
            if column_index[1] < self.feature_column_size:
                deleted_column_num += self.feature_column_size - column_index[1] + 1
            self.feature_column_size -= deleted_column_num
    
    # Drop original features
    def drop_org_features(self, inplace = False):
        # check if data is empty
        if self.column_size <= 0 or self.raw_size <= 0 or (self.data is None):
            if inplace == True:
                self.clear()
            return np.zeros((0,0))
        
        # Return dropped data
        if inplace == False:
            # Will not inplace orginal object, just return the dropped data array
            ret = self.data[:,self.feature_column_size :]
            return pd.DataFrame(ret,columns=self.columns[self.feature_column_size:])
        elif inplace == True:
            # Will replace the orginal object
            self.data = self.data[:,self.feature_column_size :]
            self.columns = self.columns[self.feature_column_size:]
            self.column_size -= self.feature_column_size
            self.feature_column_size = 0       
            return pd.DataFrame(self.data,columns=self.columns)
  
    # clear all data
    def clear(self):
        self.delete_mul_columns([0,self.column_size-1])
        self.columns.clear()
        self.feature_column_size = 0
        self.column_size = 0
        self.target = None
        self.data = None
        self.raw_size = 0
        self.data_type = None
    
    # clear data and target contents only but reserves column names
    def clear_data(self):    
        self.data = np.zeros((0,self.column_size))
        self.target = np.zeros(0)
        self.raw_size = 0
        self.data_type = None
