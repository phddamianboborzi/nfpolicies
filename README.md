# Normalizing Flow Policy implementation

Original implementation of the Normalizing Flow policy used in "Learning Normalizing Flow Policies Based on Highway Demonstrations" 
(https://ieeexplore.ieee.org/document/9564456). 

## install dependencies 
Dependencies can be found in ./requirements.txt

The standard configuration uses AntBulletEnv-v0 from pybullet (pybullet.org). 
PyBullet can be installed with:
<pre><code>
pip install pybullet
</code></pre>

It is necessary to use the correct FrEIA implementation (the branch NF-Policy from the following fork is the version used in the paper):
<pre><code>
https://github.com/FeMa42/FrEIA-NF-policy/tree/NF-Policy
</code></pre>
Otherwise training the policy will not work properly and you might experience stability issues.  

## Training and testing flow policy 
Dataset containing expert data is in ./data

Start training the policy using BC with:
<pre><code>
python train.py
</code></pre>
