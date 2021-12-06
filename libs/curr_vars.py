from os import uname
hostname = uname()[1]


prefix = "[INFO]: "

setupOpt = 1

if setupOpt == 1:
   model_opt = 1
   LOOKUP_STEP = 1
elif setupOpt == 2:
   model_opt = 2
   LOOKUP_STEP = 20



# pc_model_opt = model_opt
# pc_LOOKUP_STEP = LOOKUP_STEP

exclude_var = LOOKUP_STEP-1
# exclude_var = 0

# model_opt = 4
# LOOKUP_STEP = 1 #1 business day ahead
# LOOKUP_STEP = 5 #1 business days ahead
# LOOKUP_STEP = 10 #2 business weeks ahead
# LOOKUP_STEP = 20 #4 business weeks ahead (aka a month)

timeper = "2000" #dataset starts
print(f"{prefix}timeper = {timeper}")
print(f'{prefix}model_opt = {model_opt}')
print(f'{prefix}LOOKUP_STEP = {LOOKUP_STEP}')

if model_opt == 1:
   N_STEPS = 30
   UNITS = 40
   LAYERS = 3
elif model_opt == 2:
   N_STEPS = 60
   UNITS = 80
   LAYERS = 3
else:
   print('[ERROR]: curr_vars.py: variable model_opt is set to an invalid option... exiting')
   exit(1)

try:
   UNITS
except NameError:
   UNITS = N_STEPS

close_range = 2

#anything over 1 fails
output_steps = 1
# output_steps = 3
# LOOKUP_STEP = output_steps