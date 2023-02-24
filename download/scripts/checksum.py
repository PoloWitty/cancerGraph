# this script is used to verify the md5sum code for downloaded pubmed medline baseline files
# write error log to `error.log`

import glob
import re
import pdb
import subprocess
import tqdm

if __name__=='__main__':
    file_names = glob.glob('./pubmed*.xml.gz.md5')

    pattern = re.compile(r'MD5\((.*)\)= (.*)')
    open('error.log','w').close()
    error_log = open('error.log','a')
    for file_name in tqdm.tqdm(file_names):
        with open(file_name) as fp:
            line = fp.readline().strip()
            if line =='':
                error_log.write(file_name+'\n')
                continue

            out = pattern.findall(line)
            name = out[0][0];md5 = out[0][1]
            file_md5 = subprocess.getoutput('md5sum %s'%name).split(' ')[0]
            # pdb.set_trace()
            if file_md5!=md5:
                error_log.write(name+'\n')
