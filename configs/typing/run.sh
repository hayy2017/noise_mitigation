
trnscript=../../src/typing/train2level.py
#attention
python $trnscript --config config.yaml --train true --testmulti true --multitype attent --outtype att

#MIML-MAX
python $trnscript --config config.yaml --config config.yaml --train true --testmulti t --outtype max --multitype max

#MIML-AVG
python $trnscript  --config config.yaml --train true --testmulti t --outtype mean --multitype mean
