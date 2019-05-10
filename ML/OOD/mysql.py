#!/usr/bin/env python3
# Matt Stillwell
import pymysql.cursors


class SQLConnector(object):

    def __init__(self, host = 'localhost'):
        """ Constructor """
        # Connect to the database
        self.connection = pymysql.connect(host=host,
                                          port=3306,
                                          user='ldeng',
                                          password='cs*titanML',
                                          db='results',
                                          charset='utf8mb4',
                                          cursorclass=pymysql.cursors.DictCursor)

    def _create_hyper(self):
        """ Creates the hypers table """
        try:
            with self.connection.cursor() as cursor:
                sql = """
                        create table hypers (
                            id int, 
                            iteration int, 
                            layers varchar(255), 
                            attack int, 
                            accuracy float(20)
                        );
                    """
                cursor.execute(sql)
            self.connection.commit()
        finally:
            pass

    def write_hypers(self, theid, iteration, layersstr, attack, acc):
        """ Writes to the hypers table """

        try:
            with self.connection.cursor() as cursor:
                sql = """ 
                            insert into hypers (id, iteration, layers, attack, accuracy) 
                            values (%s, %s, %s, %s, %s); 
                        """
                cursor.execute(sql, (str(theid), str(iteration), str(layersstr), str(attack), str(acc)))

            # connection is not autocommit by default. So you must commit to save
            # your changes.
            self.connection.commit()

        finally:
            pass

    def _get_max_iter(self, modelnum):
        try:
            with self.connection.cursor() as cursor:
                sql1 = """
                            select max(hypers_id) 
                            from hypers 
                            where id = %s;
                        """

                cursor.execute(sql1, (str(modelnum)))
                dicti = cursor.fetchall()[0]
                maxnum = dicti.get('max(iteration)')
                if maxnum is None:
                    maxnum = 0
                maxnum += 1

            return maxnum

        finally:
            pass
    '''
    def write(self, gennum, modelnum, layersstr, accuracy, attack_type, gen_list):
        """ Writes to the hypers and gens table """
        iteration = self._get_max_iter(modelnum)
        self.write_hypers(modelnum, iteration, layersstr, attack_type, accuracy)
        self._write_gens(gennum, modelnum, iteration, gen_list[0], gen_list[1], gen_list[2], gen_list[3],
                         gen_list[4], gen_list[5], gen_list[6], gen_list[7], gen_list[8], gen_list[9], gen_list[10],
                         gen_list[11], gen_list[12], gen_list[13], gen_list[14], gen_list[15], gen_list[16], gen_list[17],
                         gen_list[18], gen_list[19], gen_list[20], gen_list[21], gen_list[22], gen_list[23], gen_list[24],
                         gen_list[25], gen_list[26], gen_list[27], gen_list[28], gen_list[29], gen_list[30], gen_list[31],
                         gen_list[32], gen_list[33], gen_list[34], gen_list[35], gen_list[36], gen_list[37], gen_list[38],
                         gen_list[39], gen_list[40], attack_type)
'''

    def write_hypers(self, layerstr, attack_encoded, accuracy):
        try:
            with self.connection.cursor() as cursor:
                sql = """ 
                            insert into results.hypers (hypers_id, layers, attack, accuracy) 
                            values (NULL, %s, %s, %s); 
                        """
                        # null because auto increment
                cursor.execute(sql, [str(layerstr), str(attack_encoded), str(accuracy)])

            self.connection.commit()

        finally:
            pass


    def _sort_hyper(self):
        """ Reorders the hypers table based off of id """
        try:
            with self.connection.cursor() as cursor:
                sql = """
                        alter table results.hypers 
                        order by hypers_id asc;
                    """
                cursor.execute(sql)
            self.connection.commit()
        finally:
            pass

    def read_hypers(self, acc=0):
        """ Reads from the hypers table dependent on accuracy """
        try:
            with self.connection.cursor() as cursor:
                sql = """
                        select * from hypers 
                        where accuracy > %s;
                    """
                cursor.execute(sql, (str(acc)))
                result = cursor.fetchall()
                return result
        finally:
            pass

    #=====================================================

    def _create_gens(self):
        """ Creates the gen table """
        try:
            with self.connection.cursor() as cursor:
                sql = """ 
                        create table gens (
                            gens_id int, 
                            h_id int, 
                            duration int, 
                            protocol_type varchar(10), 
                            service varchar(50), 
                            flag varchar(100), 
                            src_bytes int, 
                            dst_bytes int, 
                            land int, 
                            wrong_fragment int, 
                            urgent int, 
                            hot int, 
                            num_failed_logins int, 
                            logged_in int, 
                            num_compromised int, 
                            root_shell int, 
                            su_attempted int, 
                            num_root int, 
                            num_file_creations int, 
                            num_shells int, 
                            num_access_files int, 
                            num_outbound_cmds int, 
                            is_host_login int, 
                            is_guest_login int, 
                            count int, 
                            srv_count int, 
                            serror_rate int, 
                            srv_serror_rate int, 
                            rerror_rate int, 
                            srv_rerror_rate int, 
                            same_srv_rate int, 
                            diff_srv_rate int, 
                            srv_diff_host_rate int, 
                            dst_host_count int, 
                            dst_host_srv_count int, 
                            dst_host_same_srv_rate int, 
                            dst_host_diff_srv_rate float(20), 
                            dst_host_same_src_port_rate float(20), 
                            dst_host_srv_diff_host_rate int, 
                            dst_host_serror_rate int, 
                            dst_host_srv_serror_rate int, 
                            dst_host_rerror_rate int, 
                            dst_host_srv_rerror_rate int, 
                            attack_type int
                        );
                    """
                cursor.execute(sql)
            self.connection.commit()
        finally:
            pass

    def write_gens(self, gen_attack_array, attack_type):
        hypers_id = self.get_last_model_id()[0]
        hypers_id = hypers_id['max(hypers_id)']
        print(type(hypers_id))
        print(hypers_id)
        gen_attack_array = gen_attack_array.astype(int)
        print(len(gen_attack_array[0]))
        print(gen_attack_array[0, :])
        for i in range(0, len(gen_attack_array[:, 0])):
            self._write_gen_attack(hypers_id, gen_attack_array[i, 0], gen_attack_array[i, 1], gen_attack_array[i, 2],
                                   gen_attack_array[i, 3], gen_attack_array[i, 4], gen_attack_array[i, 5],
                                   gen_attack_array[i, 6], gen_attack_array[i, 7], gen_attack_array[i, 8],
                                   gen_attack_array[i, 9], gen_attack_array[i, 10], gen_attack_array[i, 11],
                                   gen_attack_array[i, 12], gen_attack_array[i, 13], gen_attack_array[i, 14],
                                   gen_attack_array[i, 15], gen_attack_array[i, 16], gen_attack_array[i, 17],
                                   gen_attack_array[i, 18], gen_attack_array[i, 19], gen_attack_array[i, 20],
                                   gen_attack_array[i, 21], gen_attack_array[i, 22], gen_attack_array[i, 23],
                                   gen_attack_array[i, 24], gen_attack_array[i, 25], gen_attack_array[i, 26],
                                   gen_attack_array[i, 27], gen_attack_array[i, 28], gen_attack_array[i, 29],
                                   gen_attack_array[i, 30], gen_attack_array[i, 31], gen_attack_array[i, 32],
                                   gen_attack_array[i, 33], gen_attack_array[i, 34], gen_attack_array[i, 35],
                                   gen_attack_array[i, 36], gen_attack_array[i, 37], gen_attack_array[i, 38],
                                   gen_attack_array[i, 39], gen_attack_array[i, 40], attack_type)



    def _write_gen_attack(self, h_id, duration, protocol_type, service, flag,
                    src_bytes, dst_bytes, land, wrong_fragment, urgent, hot, num_failed_logins, logged_in,
                    num_compromised, root_shell, su_attempted, num_root, num_file_creations, num_shells,
                    num_access_files, num_outbound_cmds, is_host_login, is_guest_login, count, srv_count,
                    serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate, same_srv_rate,
                    diff_srv_rate, srv_diff_host_rate, dst_host_count, dst_host_srv_count,
                    dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate,
                    dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate,
                    dst_host_rerror_rate, dst_host_srv_rerror_rate, attack_type):
        """ Writes to the gens table """

        try:
            with self.connection.cursor() as cursor:
                sql = """
                        insert into results.gens ( 
                            gens_id,
                            h_id, 
                            duration, 
                            protocol_type, 
                            service, 
                            flag, 
                            src_bytes, 
                            dst_bytes, 
                            land, 
                            wrong_fragment, 
                            urgent, 
                            hot, 
                            num_failed_logins, 
                            logged_in, 
                            num_compromised, 
                            root_shell, 
                            su_attempted, 
                            num_root, 
                            num_file_creations, 
                            num_shells, 
                            num_access_files, 
                            num_outbound_cmds, 
                            is_host_login, 
                            is_guest_login, 
                            cnt, 
                            srv_count, 
                            serror_rate, 
                            srv_serror_rate, 
                            rerror_rate, 
                            srv_rerror_rate, 
                            same_srv_rate, 
                            diff_srv_rate, 
                            srv_diff_host_rate, 
                            dst_host_count, 
                            dst_host_srv_count, 
                            dst_host_same_srv_rate, 
                            dst_host_diff_srv_rate, 
                            dst_host_same_src_port_rate, 
                            dst_host_srv_diff_host_rate, 
                            dst_host_serror_rate, 
                            dst_host_srv_serror_rate, 
                            dst_host_rerror_rate, 
                            dst_host_srv_rerror_rate, 
                            attack_type
                        ) values (
                            NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                            %s, %s, %s, %s
                        );
                    """

                cursor.execute(sql, (str(h_id), str(duration),
                                     str(protocol_type), str(service), str(flag), str(src_bytes),
                                     str(dst_bytes), str(land), str(wrong_fragment), str(urgent),
                                     str(hot), str(num_failed_logins), str(logged_in),
                                     str(num_compromised), str(root_shell), str(su_attempted),
                                     str(num_root), str(num_file_creations), str(num_shells),
                                     str(num_access_files), str(num_outbound_cmds),
                                     str(is_host_login), str(is_guest_login), str(count),
                                     str(srv_count), str(serror_rate), str(srv_serror_rate),
                                     str(rerror_rate), str(srv_rerror_rate), str(same_srv_rate),
                                     str(diff_srv_rate), str(srv_diff_host_rate),
                                     str(dst_host_count), str(dst_host_srv_count),
                                     str(dst_host_same_srv_rate), str(dst_host_diff_srv_rate),
                                     str(dst_host_same_src_port_rate),
                                     str(dst_host_srv_diff_host_rate), str(dst_host_serror_rate),
                                     str(dst_host_srv_serror_rate), str(dst_host_rerror_rate),
                                     str(dst_host_srv_rerror_rate), str(attack_type)))

                # connection is not autocommit by default. So you must commit to save
            # your changes.
            self.connection.commit()

        finally:
            pass

    def read_gens(self):
        """ Reads from the gens table dependent on accuracy """
        try:
            with self.connection.cursor() as cursor:
                sql = """
                        select gens_id, h_id, attack_type 
                        from gens;
                    """ #Does not return correct output for accuracies
                cursor.execute(sql)
                result = cursor.fetchall()
                return result
        finally:
            pass

    #=============================================

    def _read_specific_joined(self, theid, theiter):
        """ Reads the joined table at a specific id and iteration """
        try:
            with self.connection.cursor() as cursor:
                sql = """
                        select gens.id, modelnum, gens.iteration, layers, attack_type, name, accuracy 
                        from gens, hypers, attacks
                        where gens.modelnum = hypers.id 
                        and gens.iteration = hypers.iteration
                        and hypers.attack = attacks.id
                        and gens.id = %s 
                        and gens.iteration = %s;
                    """
                cursor.execute(sql, (str(theid), str(theiter)))
                result = cursor.fetchall()
                return result
        finally:
            pass

    def _read_joined(self, acc=0):
        """ Reads the joined table dependent on the accuracy """
        try:
            with self.connection.cursor() as cursor:
                sql = """
                        select gens.id, modelnum, gens.iteration, layers, attack_type, name, accuracy 
                        from gens, hypers, attacks 
                        where gens.modelnum = hypers.id 
                        and gens.iteration = hypers.iteration 
                        and hypers.attack = attacks.id 
                        and accuracy > %s;
                    """
                cursor.execute(sql, (str(acc)))
                result = cursor.fetchall()
                return result
        finally:
            pass

    def _use_datasets(self):
        """ Select datasets database """
        try:
            with self.connection.cursor() as cursor:
                sql = "use datasets;"
                cursor.execute(sql)
                result = cursor.fetchall()
                return result
        finally:
            pass

    def _use_results(self):
        """ Select results database """
        try:
            with self.connection.cursor() as cursor:
                sql = "use results;"
                cursor.execute(sql)
                result = cursor.fetchall()
                return result
        finally:
            pass


    def pull_best_results(self, attack, num=1, verbose=False):
        """ Reads the joined table at a specific id and iteration """
        self._use_results()
        try:
            with self.connection.cursor() as cursor:
                if not verbose:
                    sql = """
                            select *
                            from hypers, attacks
                            where hypers.attack = attacks.id
                            and name = %s
                            order by accuracy desc
                            limit %s;
                        """
                else:
                    sql = """
                            select *
                            from gens, hypers, attacks
                            where gens.modelnum = hypers.id
                            and gens.iteration = hypers.iteration
                            and hypers.attack = attacks.id
                            and name = %s
                            order by accuracy desc
                            limit %s;
                        """
                cursor.execute(sql, (str(attack), num))
                result = cursor.fetchall()
                return result
        finally:
            pass

    def pull_kdd99(self, attack, num, nodupes = False):
        """ Returns randomly shuffled data by attack type """
        self._use_datasets()
        try:
            with self.connection.cursor() as cursor:
                if nodupes:
                    sql = """
                            select *
                            from kdd99_dupless
                            where attack_type like %s
                            order by RAND ()
                            limit %s;
                        """
                else:
                    sql = """
                            select *
                            from kdd99
                            where attack_type like %s
                            order by RAND ()
                            limit %s;
                        """
                cursor.execute(sql, ('%'+str(attack)+'%', num))
                result = cursor.fetchall()
                return result
        finally:
            pass
    
    def pull_all_attacks(self, num, nodupes = False):
        """ Returns randomly shuffled data by attack type """
        self._use_datasets()
        try:
            with self.connection.cursor() as cursor:
                if nodupes:
                    sql = """
                            select *
                            from kdd99_dupless
                            order by RAND ()
                            limit %s;
                        """
                else:
                    sql = """
                            select *
                            from kdd99
                            order by RAND ()
                            limit %s;
                        """
                cursor.execute(sql, num)
                result = cursor.fetchall()
                return result
        finally:
            pass
        
    def pull_evaluator_data(self, num, attack):
        # We want equal attack and non-attack data, this handles the case that num is an odd number
        first_half = int(num / 2)
        second_half = num - first_half

        self._use_datasets()
        try:
            with self.connection.cursor() as cursor:
                sql = """
                       (select * from kdd99
                       where attack_type like %s
                       order by RAND ()
                       limit %s)
                       union all
                       (select * from kdd99
                       where attack_type not like %s
                       order by RAND ()
                       limit %s);
                   """
                cursor.execute(sql, ('%'+str(attack)+'%', first_half, '%'+'normal'+'%', second_half))
                result = cursor.fetchall()
                return result
        finally:
            pass

    @staticmethod
    def pull_kdd99_columns(allQ=True):
        """ Returns kdd99 col names """
        ls = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
              "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
              "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
              "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
              "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
              "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
              "dst_host_srv_rerror_rate"]
        if(allQ):
            ls.append("attack_type")
        return ls

    #====================================================

    def _create_attacks(self):
        """ Creates the hypers table """
        try:
            with self.connection.cursor() as cursor:
                sql = """
                        create table attacks (
                            id int,
                            name varchar(255)
                        );
                    """
                cursor.execute(sql)
            self.connection.commit()
        finally:
            pass

    def pull_evaluator_data(self, num, attack):
        # We want equal attack and non-attack data, this handles the case that num is an odd number
        first_half = int(num / 2)
        second_half = num - first_half

        self._use_datasets()
        try:
            with self.connection.cursor() as cursor:
                sql = """
                        (select * from kdd99 
                        where attack_type like %s
                        order by RAND ()
                        limit %s)
                        union all
                        (select * from kdd99
                        where attack_type not like %s
                        order by RAND ()
                        limit %s);
                    """
                cursor.execute(sql, ('%'+str(attack)+'%', first_half, '%'+str(attack)+'%', second_half))
                result = cursor.fetchall()
                return result
        finally:
            pass

    def _write_attacks(self, theid, attack):
        """ Writes to the attacks table """
        try:
            with self.connection.cursor() as cursor:
                sql = """
                        insert into attacks (id, attack_name)
                        values (%s, %s);
                    """
                cursor.execute(sql, (str(theid), str(attack)))
            # connection is not autocommit by default. So you must commit to save
            # your changes.
            self.connection.commit()
        finally:
            pass

    def get_last_model_id(self):
        self._use_datasets()
        try:
            with self.connection.cursor() as cursor:
                sql = """
                        select max(hypers_id)
                        from results.hypers;
                            """
                cursor.execute(sql)
                result = cursor.fetchall()
                return result
        finally:
            pass

    def _fill_attacks(self):
        self._write_attacks(1, "normal")
        self._write_attacks(2, "buffer_overflow")
        self._write_attacks(3, "loadmodule")
        self._write_attacks(4, "perl")
        self._write_attacks(5, "neptune")
        self._write_attacks(6, "smurf")
        self._write_attacks(7, "guess_passwd")
        self._write_attacks(8, "pod")
        self._write_attacks(9, "teardrop")
        self._write_attacks(10, "portsweep")
        self._write_attacks(11, "ipsweep")
        self._write_attacks(12, "land")
        self._write_attacks(13, "ftp_write")
        self._write_attacks(14, "back")
        self._write_attacks(15, "imap")
        self._write_attacks(16, "satan")
        self._write_attacks(17, "phf")
        self._write_attacks(18, "nmap")
        self._write_attacks(19, "multihop")
        self._write_attacks(20, "warezmaster")
        self._write_attacks(21, "warezclient")
        self._write_attacks(22, "spy")
        self._write_attacks(23, "rootkit")


    def _read_attacks(self):
        """ Reads from the attack table dependent on num """
        try:
            with self.connection.cursor() as cursor:
                sql = "select id, name from attacks;"
                cursor.execute(sql)
                result = cursor.fetchall()
                return result
        finally:
            pass


def main():
    """ Auto run main method """
    conn = SQLConnector()

    # print(conn.pull_best_results("neptune"), 5, True)
    #print(conn.pull_best_results("neptune"))

    #print(conn._read_joined())
    #print(conn._read_specific_joined(1,1))

    #conn._create_gens()
    #conn._create_hyper()

    #conn._create_attacks()
    #conn._fill_attacks()


    conn.write_hypers("2,3,4", 5, 20.3)
    conn.write_gens(592, 0, "tcp", "ftp_data", "REJ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0.00, 171, 62, 0.27, 0.02, 0.01, 0.03, 0.01, 0, 0.29, 0.02, 10)

    print(conn.read_gens())
    print(conn.read_hypers())

    #conn._sort_hyper()

    #print(conn._read_specific_joined(1, 1))
    #print(conn._read_joined(20))
    #conn._output_to_csv(conn.read_joined(20))

if __name__ == "__main__":
    main()

