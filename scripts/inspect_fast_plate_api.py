import inspect
from fast_plate_ocr.inference.plate_recognizer import LicensePlateRecognizer
print('SIGNATURE:', inspect.signature(LicensePlateRecognizer))
print('INIT SOURCE:')
print(inspect.getsource(LicensePlateRecognizer.__init__))
print('--- METHODS ---')
for name, member in inspect.getmembers(LicensePlateRecognizer, predicate=inspect.isfunction):
    if not name.startswith('_'):
        print(name, inspect.signature(member))
