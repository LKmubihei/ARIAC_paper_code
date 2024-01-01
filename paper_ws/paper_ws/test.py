import re
text_generic = "agv2"
numbers = re.findall('agv(\d+)', text_generic)
agv_numbers = int(numbers[0])
print(agv_numbers,type(agv_numbers))