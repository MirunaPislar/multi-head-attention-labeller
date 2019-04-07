import re

string = '''
           O     0.2660    0.3419    0.2992       389
           N     0.7213    0.6414    0.6790       912
           P     0.7675    0.7591    0.7633       909

   micro avg     0.6371    0.6371    0.6371      2210
   macro avg     0.5850    0.5808    0.5805      2210
   
           O     0.7520    0.2515    0.3769     31998
           N     0.0809    0.1883    0.1132      3760
           P     0.1631    0.5631    0.2529      6647

   micro avg     0.2947    0.2947    0.2947     42405
   macro avg     0.3320    0.3343    0.2477     42405
'''

splits = string.split("\n")
for s in splits:
  if len(s) == 0:
    continue
  spl = re.split(r'\s{2,}', s)
  print("\t".join(spl[2:-1]))


