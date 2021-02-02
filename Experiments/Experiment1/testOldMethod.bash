file=$1
mgs=$2
maxImg=$3
k=$4
accGoal=$5
python3 -W ignore OldMethod.py $file $mgs knndatasetGI $maxImg $k $accGoal
