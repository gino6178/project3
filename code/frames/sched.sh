#!/bin/bash
# dynamic scheduler: one worker per GPU pulls the next object from ~/ov/queue.txt (mkdir is the lock)
worker(){ g=$1; while true; do
  until mkdir ~/ov/.lock 2>/dev/null; do sleep 1; done
  o=$(head -n 1 ~/ov/queue.txt); sed -i '1d' ~/ov/queue.txt; rmdir ~/ov/.lock
  [ -z "$o" ] && break
  echo "$(date +%H:%M) gpu$g $o" >> ~/ov/sched.log; bash ~/ovobj.sh $o $g; echo "$(date +%H:%M) gpu$g $o done" >> ~/ov/sched.log
done; }
for g in 0 1 2 3; do worker $g & done; wait; echo SCHED_DONE >> ~/ov/sched.log
