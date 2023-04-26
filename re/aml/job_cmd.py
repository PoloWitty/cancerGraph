"""
desc:	write job cmd
author:	Yangzhe Peng
date:	2023/04/06
"""

with open('jobs.txt','w') as fp:
    for i in range(48):
        fp.write('''
  - name: cancerRE-{0}
    sku: 1x G1
    command:
    - bash run.sh {0} 8
        '''.format(i))

print('done!')