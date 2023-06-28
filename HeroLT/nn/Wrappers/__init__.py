try: 
    from .BaseModel import BaseModel
except Exception as e:
    print('File to load BaseModel. Packages need to be installed. Please reload the framework after installing the environment')
    raise RuntimeError(e)

## Graph Models
errors = []
packages = []
try:
    from .GraphSMOTE import GraphSMOTE
except Exception as e:
    errors.append('File to load GraphSMOTE. Packages need to be installed for GraphSMOTE. Please reload the framework after installing the environment')
    packages.append(e)
try:
    from .ImGAGN import ImGAGN
except Exception as e:
    errors.append('File to load ImGAGN. Packages need to be installed for ImGAGN. Please reload the framework after installing the environment')
    packages.append(e)
try:
    from .LTE4G import LTE4G
except Exception as e:
    errors.append('File to load LTE4G. Packages need to be installed for LTE4G. Please reload the framework after installing the environment')
    packages.append(e)
try:
    from .TailGNN import TailGNN
except Exception as e:
    errors.append('File to load TailGNN. Packages need to be installed for TailGNN. Please reload the framework after installing the environment')
    packages.append(e)

if len(errors) > 0:
    print('----------------------------Fail to load Graph Models----------------------------')
    for s, e in zip(errors, packages):
        print(e)
        print(s)
        print('\n')

## CV Models
errors = []
packages = []
try:
    from .CVModel import CVModel
except Exception as e:
    errors.append('File to load CVModel. Packages need to be installed. BALMS and Decoupling can\'t run under this situation. Please reload the framework after installing the environment')
    packages.append(e)
try:
    from .BALMS import BALMS
except Exception as e:
    errors.append('File to load BALMS. Packages need to be installed for BALMS. Please reload the framework after installing the environment')
    packages.append(e)
try:
    from .BBN import BBN
except Exception as e:
    errors.append('File to load BBN. Packages need to be installed for BBN. Please reload the framework after installing the environment')
    packages.append(e)
try:
    from .Decoupling import Decoupling
except Exception as e:
    errors.append('File to load Decoupling. Packages need to be installed for Decoupling. Please reload the framework after installing the environment')
    packages.append(e)
try:
    from .MiSLAS import MiSLAS
except Exception as e:
    errors.append('File to load MiSLAS. Packages need to be installed for MiSLAS. Please reload the framework after installing the environment')
    packages.append(e)
try:
    from .OLTR import OLTR
except Exception as e:
    errors.append('File to load OLTR. Packages need to be installed for OLTR. Please reload the framework after installing the environment')
    packages.append(e)
try:
    from .TDE import TDE
except Exception as e:
    errors.append('File to load TDE. Packages need to be installed for TDE. Please reload the framework after installing the environment')
    packages.append(e)

if len(errors) > 0:
    print('----------------------------Fail to load CV Models----------------------------')
    for s, e in zip(errors, packages):
        print(e)
        print(s)
        print('\n')

## NLP Models
errors = []
packages = []
try:
    from .XRLinear import XRLinear
except Exception as e:
    errors.append('File to load XRLinear. Packages need to be installed for XRLinear. Please reload the framework after installing the environment')
    packages.append(e)
try:
    from .XRTransformer import XRTransformer
except Exception as e:
    errors.append('File to load XRTransformer. Packages need to be installed for XRTransformer. Please reload the framework after installing the environment')
    packages.append(e)
try:
    from .XTransformer import XTransformer
except Exception as e:
    errors.append('File to load XTransformer. Packages need to be installed for XTransformer. Please reload the framework after installing the environment')
    packages.append(e)

if len(errors) > 0:
    print('----------------------------Fail to load NLP Models----------------------------')
    for s, e in zip(errors, packages):
        print(e)
        print(s)
        print('\n')

