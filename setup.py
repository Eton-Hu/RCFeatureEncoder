from distutils.core import setup

#installation script

setup(
    name= 'RCFeatureEncoder',
    version= '1.0.0',
    py_modules =['RCFeatureEncoder','RCFeatureEncoder.RCEncodedForm','RCFeatureEncoder.RCREModule'],
    author= 'Hu Yitong',
    author_email='yitong.hu@outlook.com',
    url='1',
    description= 'A mean/target encoder tool for preprocessing data before using learning algorithms',
    install_requires=[
                        'sklearn',
                        'numpy',
                        'pandas'
                    ]
    )