# Putting Conda on your own computer

## Creating conda env
conda env create -f conda-ml-environment.yml

## To activate env
conda activate ml

## To deactivate 
source deactivate 



# MySql

## Creating kdd99 table
create table kdd99 (duration int, protocol_type varchar(10), service varchar(50), flag varchar(100), src_bytes int, dst_bytes int, land int, wrong_fragment int, urgent int, hot int, num_failed_logins int, logged_in int, num_compromised int, root_shell int, su_attempted int, num_root int, num_file_creations int, num_shells int, num_access_files int, num_outbound_cmds int, is_host_login int, is_guest_login int, count int, srv_count int, serror_rate int, srv_serror_rate int, rerror_rate int, srv_rerror_rate int, same_srv_rate int, diff_srv_rate int, srv_diff_host_rate int, dst_host_count int, dst_host_srv_count int, dst_host_same_srv_rate int, dst_host_diff_srv_rate float(20), dst_host_same_src_port_rate float(20), dst_host_srv_diff_host_rate int, dst_host_serror_rate int, dst_host_srv_serror_rate int, dst_host_rerror_rate int, dst_host_srv_rerror_rate int, attack_type varchar(50));

## Move dataset to be read in
Move into /var/lib/mysql/datasets/kdd10.csv

## Import csv into database
load data infile 'kdd10.csv' into table datasets.kdd99 fields terminated by ',';

