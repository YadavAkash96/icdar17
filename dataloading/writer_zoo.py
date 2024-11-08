import os.path
from .regex import ImageFolder

class WriterZoo:

    @staticmethod
    def new(desc, **kwargs):
        return ImageFolder(desc['path'], regex=desc['regex'], **kwargs)

    @staticmethod
    def get(dataset, set, **kwargs):
        _all = WriterZoo.datasets
        d = _all[dataset]
        s = d['set'][set]
        print(s['path'])
        s['path'] = os.path.join(d['basepath'], s['path'])
        print(f"s=>>>>>{s}")
        return WriterZoo.new(s, **kwargs)

    datasets = {

        'icdar2017': {
            'basepath': r'/home/vault/iwi5/iwi5232h/resources/datasets/',
            'set': {
                'test' :  {'path': 'scriptnet',
                                  'regex' : {'writer': r'(\d+)', 'page': r'\d+-IMG_MAX_(\d+)'}},

                'train' :  {'path': 'icdar17',
                                  'regex' : {'cluster' : r'(\d+)', 'writer': r'\d+_(\d+)', 'page' : r'\d+_\d+-\d+-IMG_MAX_(\d+)'}},
                
            }
        },

        'icdar2013': {
            'basepath': '/data/mpeer/resources',
            'set': {
                'test' :  {'path': r'icdar2013_test_sift_patches_binarized',
                                  'regex' : {'writer': r'(\d+)', 'page': r'\d+_(\d+)'}},

                'train' :  {'path': 'icdar2013_train_sift_patches_1000/',
                                  'regex' : {'cluster' : r'(\d+)', 'writer': r'\d+_(\d+)', 'page' : r'\d+_\d+_(\d+)'}}                             
            }
        },
        
        'icdar2019': {
            'basepath': r'/data/mpeer/resources',
            'set': {
                'test' :  {'path': r'wi_comp_19_test_patches',
                                  'regex' : {'writer': r'(\d+)', 'page': r'\d+_(\d+)'}},

                'train' :  {'path': r'wi_comp_19_validation_patches',
                                  'regex' : {'cluster' : r'(\d+)', 'writer': r'\d+_(\d+)', 'page' : r'\d+_\d+_(\d+)'}},
            }
        }
    }