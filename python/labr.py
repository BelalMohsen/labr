# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:27:03 2013

@author: Mohamed Aly <mohamed@mohamedaly.info>
"""

import codecs
import numpy as np
import pandas as pd
import re

class LABR:
    def __init__(self):
        self.REVIEWS_PATH = "../data/labr_data/"
        self.RAW_REVIEWS_FILE = "raw_reviews.tsv"
        self.DELETED_REVIEWS_FILE = "deleted_reviews.tsv"
        self.CLEAN_REVIEWS_FILE = "reviews.tsv"
        self.CLEAN_NO_STOP_REVIEWS_FILE = "nostopwords_reviews"
        self.CLEAN_NO_STOP_STEMMED_REVIEWS_FILE = "nostopwords_stemmed_reviews"
        self.NORMALIZED_REVIEWS_FILE = "norm_reviews.tsv"
    # Copied from the PyArabic package.
    def arabicrange(self):
        """return a list of arabic characteres .
        Return a list of characteres between \u060c to \u0652
        @return: list of arabic characteres.
        @rtype: unicode;
        """
        mylist = [];
        for i in range(0x0600, 0x00653):
            try :
                mylist.append(unichr(i));
            except ValueError:
                pass;
        return mylist;

    # cleans a single review
    def clean_raw_review(self, body):
         # patterns to remove first
        pat = [\
            (u'http[s]?://[a-zA-Z0-9_\-./~\?=%&]+', u''),  # remove links
            (u'www[a-zA-Z0-9_\-?=%&/.~]+', u''),
#            u'\n+': u' ',                     # remove newlines
            (u'<br />', u' '),  # remove html line breaks
            (u'</?[^>]+>', u' '),  # remove html markup
#            u'http': u'',
            (u'[a-zA-Z]+\.org', u''),
            (u'[a-zA-Z]+\.com', u''),
            (u'://', u''),
            (u'&[^;]+;', u' '),
            (u':D', u':)'),
#            (u'[0-9/]+', u''),
#            u'[a-zA-Z.]+': u'',
#            u'[^0-9' + u''.join(self.arabicrange()) + \
#                u"!.,;:$%&*%'#(){}~`\[\]/\\\\\"" + \
#                u'\s^><\-_\u201D\u00AB=\u2026]+': u'',          # remove latin characters
            (u'\s+', u' '),  # remove spaces
            (u'\.+', u'.'),  # multiple dots
            (u'[\u201C\u201D]', u'"'),  # “
            (u'[\u2665\u2764]', u''),  # heart symbol
            (u'[\u00BB\u00AB]', u'"'),
            (u'\u2013', u'-'),  # dash
        ]

        # patterns that disqualify a review
        remove_if_there = [\
            (u'[^0-9' + u''.join(self.arabicrange()) + \
                u"!.,;:$%&*%'#(){}~`\[\]/\\\\\"" + \
                u'\s\^><\-_\u201D\u00AB=\u2026+|' + \
                u'\u0660-\u066D\u201C\u201D' + \
                u'\ufefb\ufef7\ufef5\ufef9]+', u''),  # non arabic characters
        ]

        # patterns that disqualify if empty after removing
        remove_if_empty_after = [\
            (u'[0-9a-zA-Z\-_]', u' '),  # alpha-numeric
            (u'[0-9' + u".,!;:$%&*%'#(){}~`\[\]/\\\\\"" + \
                u'\s\^><`\-=_+]+', u''),  # remove just punctuation
            (u'\s+', u' '),  # remove spaces
        ]

        # remove again
        # patterns to remove
        pat2 = [\
#            u'[^0-9' + u''.join(self.arabicrange()) + \
#                u"!.,;:$%&*%'#(){}~`\[\]/\\\\\"" + \
#                u'\s^><\-_\u201D\u00AB=\u2026]+': u'',          # remove latin characters
        ]

        skip = False

        # if empty body, skip
        if body == u'': skip = True

        # do some subsitutions
        for k, v in pat:
            body = re.sub(k, v, body)

        # remove if exist
        for k, v in remove_if_there:
            if re.search(k, body):
                skip = True

        # remove if empty after replacing
        for k, v in remove_if_empty_after:
            temp = re.sub(k, v, body)
            if temp == u" " or temp == u"":
                skip = True

        # do some more subsitutions
        if not skip:
            for k, v in pat2:
                body = re.sub(k, v, body)

        # if empty string, skip
        if body == u'' or body == u' ':
            skip = True

        if not skip:
            return body
        else:
            return u""

    # Read raw reviews from file and clean and write into clean_reviews
    def clean_raw_reviews(self):
        # input file
        in_file = codecs.open(self.REVIEWS_PATH + self.RAW_REVIEWS_FILE,
                              'r', encoding="utf-8")
        reviews = in_file.readlines()

        # Output file: rating<tab>content
        out_file = open(self.REVIEWS_PATH + self.CLEAN_REVIEWS_FILE,
                        'w', buffering=100)
        deleted_file = open(self.REVIEWS_PATH + self.DELETED_REVIEWS_FILE,
                            'w', buffering=100)

        counter = 1
        for i in xrange(0, len(reviews)):
            review = reviews[i]
            skip = False

#           # If line starts with #, then skip
#            if review[0] == u"#": continue

            # split by <tab>
            parts = review.split(u"\t")

            # rating is first part and body is last part
            rating = parts[0]
            review_id = parts[1]
            user_id = parts[2]
            book_id = parts[3]
            body = parts[4].strip()

            # clean body
            body = self.clean_raw_review(body)
            if body == u"": skip = True

            if i % 5000 == 0:
                print "review %d:" % (i)

            # write output
            line = u"%s\t%s\t%s\t%s\t%s\n" % (rating, review_id, user_id,
                                              book_id, body)
            if not skip:
                out_file.write(line.encode('utf-8'))
                counter += 1
            else:
                deleted_file.write(line.encode('utf-8'))
                
                
    

    # Read the reviews file. Returns a tuple containing these lists:
    #   rating: the rating 1 -> 5
    #   review_id: the id of the review
    #   user_id: the id of the user
    #   book_id: the id of the book
    #   body: the text of the review
    def read_review_file(self, file_name):
        reviews = codecs.open(file_name, 'r', 'utf-8').readlines()

        # remove comment lines and newlines
        reviews = [r.strip() for r in reviews if r[0] != u'#']

        # parse
        rating = list()
        review_id = list()
        user_id = list()
        book_id = list()
        body = list()
        for review in reviews:
            # split by <tab>
            parts = review.split(u"\t")

            # rating is first part and body is last part
            rating.append(int(parts[0]))
            review_id.append(parts[1])
            user_id.append(parts[2])
            book_id.append(parts[3])
            if len(parts) > 4:
                body.append(parts[4])
            else:
                body.append(u"")

        return (rating, review_id, user_id, book_id, body)

    # Writes reviews to a file
    def write_review_file(self, file_name, rating, review_id, user_id,
                          book_id, body):

        lines = list()
        # loop
        for i in xrange(len(rating)):
            line = u"%s\t%s\t%s\t%s\t%s\n" % (rating[i], review_id[i],
                                              user_id[i], book_id[i],
                                              body[i])
            lines.append(line)

        open(file_name, 'w').write(u''.join(lines).encode('utf-8'))

    def read_clean_reviews(self):
        return self.read_review_file(self.REVIEWS_PATH + 
                                     self.CLEAN_REVIEWS_FILE)

    def read_raw_reviews(self):
        return self.read_review_file(self.REVIEWS_PATH + self.RAW_REVIEWS_FILE)

    # Splits the dataset into a training/test sets in the setting of using 5
    # classes (predicting the rating value from 1 to 5)
    def split_train_test_5class(self, rating, percent_test,
                                balanced="unbalanced"):
        np.random.seed(1234)

        num_reviews = len(rating)
        review_ids = np.arange(0, num_reviews)

        if balanced == "unbalanced":
            ntest = np.floor(num_reviews * percent_test)
            np.random.shuffle(review_ids)

            test_ids = review_ids[:ntest]
            train_ids = review_ids[ntest:]

        elif balanced == "balanced":
            (sizes, bins) = np.histogram(rating, [1, 2, 3, 4, 5, 6])
            min_size = np.min(sizes)
            print min_size

            # sample review ids equally among classes
            test_ids = np.zeros((0,), dtype="int32")
            train_ids = np.zeros((0,), dtype="int32")
            rating = np.array(rating)
            ntest = np.floor(min_size * percent_test)
            for c in range(1, 6):
                cids = review_ids[np.nonzero(rating == c)]
                np.random.shuffle(cids)

                test_ids = np.r_[test_ids, cids[:ntest]]
                train_ids = np.r_[train_ids, cids[ntest:min_size]]

        train_file = self.REVIEWS_PATH + "5class-" + balanced + "-train.txt"
        test_file = self.REVIEWS_PATH + "5class-" + balanced + "-test.txt"

        open(train_file, 'w').write('\n'.join(map(str, train_ids)))
        open(test_file, 'w').write('\n'.join(map(str, test_ids)))

        return (train_ids, test_ids)

    # Splits the dataset into a training/test sets in the setting of using 2
    # classes (predicting the polarity of the review where ratings 1 & 2
    # are considered negative, ratings 4 & 5 are positive, and rating 3 is
    # ignored)
    def split_train_test_2class(self, rating, percent_test,
                                balanced="unbalanced"):
        np.random.seed(1234)

        rating = np.array(rating, dtype='int32')
        # length
        num_reviews = len(rating)
        review_ids = np.arange(0, num_reviews)

        # convert to binary, with ratings [1, 2] --> neg  and [4, 5] --> pos
        rating[rating == 2] = 1
        rating[rating == 4] = 5

        ids = (rating == 1) + (rating == 5)
        review_ids = review_ids[ids]
        rating = rating[ids]
        rating[rating == 1] = 0
        rating[rating == 5] = 1

        # get length after filtering
        num_reviews = rating.shape[0]

        if balanced == "unbalanced":
            ntest = np.floor(num_reviews * percent_test)
            np.random.shuffle(review_ids)

            test_ids = review_ids[:ntest]
            train_ids = review_ids[ntest:]

        elif balanced == "balanced":
            (sizes, bins) = np.histogram(rating, [0, 1, 2])
            min_size = np.min(sizes)
            print min_size

            # sample review ids equally among classes
            test_ids = np.zeros((0,), dtype="int32")
            train_ids = np.zeros((0,), dtype="int32")
            rating = np.array(rating)
            ntest = np.floor(min_size * percent_test)
            for c in [0, 1]:
                cids = review_ids[np.nonzero(rating == c)]
                np.random.shuffle(cids)

                test_ids = np.r_[test_ids, cids[:ntest]]
                train_ids = np.r_[train_ids, cids[ntest:min_size]]

        train_file = self.REVIEWS_PATH + "2class-" + balanced + "-train.txt"
        test_file = self.REVIEWS_PATH + "2class-" + balanced + "-test.txt"

        open(train_file, 'w').write('\n'.join(map(str, train_ids)))
        open(test_file, 'w').write('\n'.join(map(str, test_ids)))

        return (train_ids, test_ids)
    # Splits the dataset into a training/validation/test sets in the setting of using 3
    # classes (predicting the polarity of the review where ratings 1 & 2
    # are considered negative, ratings 4 & 5 are positive, and rating 3 is considered
    # neutral
    def split_train_validation_test_3class(self, rating, percent_test, percent_valid,
                                balanced="unbalanced"):
        np.random.seed(1234)

        rating = np.array(rating, dtype='int32')
        # length
        num_reviews = len(rating)
        review_ids = np.arange(0, num_reviews)

        # convert to binary, with ratings [1, 2] --> neg  and [4, 5] --> pos
        rating[rating == 2] = 1
        rating[rating == 4] = 5

        ids = (rating == 1) + (rating == 5) + (rating == 3)
        review_ids = review_ids[ids]
        rating = rating[ids]
        rating[rating == 1] = 0
        rating[rating == 5] = 1
        rating[rating == 3] = 2
        # get length after filtering
        num_reviews = rating.shape[0]

        if balanced == "unbalanced":
            ntest = np.floor(num_reviews * percent_test)
            nvalid = np.floor(num_reviews * percent_valid)
            np.random.shuffle(review_ids)

            test_ids = review_ids[:ntest]
            validation_ids = review_ids[ntest:ntest + nvalid]
            train_ids = review_ids[ntest + nvalid:]

        elif balanced == "balanced":
            (sizes, bins) = np.histogram(rating, [0, 1, 2, 3])
            min_size = np.min(sizes)
            print min_size

            # sample review ids equally among classes
            test_ids = np.zeros((0,), dtype="int32")
            validation_ids = np.zeros((0,), dtype="int32")
            train_ids = np.zeros((0,), dtype="int32")
            rating = np.array(rating)
            ntest = np.floor(min_size * percent_test)
            nvalid = np.floor(min_size * percent_valid)
            for c in [0, 1, 2]:
                cids = review_ids[np.nonzero(rating == c)]
                np.random.shuffle(cids)
                
                test_ids = np.r_[test_ids, cids[:ntest]]
                validation_ids = np.r_[validation_ids, cids[ntest:ntest + nvalid]]
                train_ids = np.r_[train_ids, cids[ntest + nvalid:min_size]]

        train_file = self.REVIEWS_PATH + "3class-" + balanced + "-train.txt"
        test_file = self.REVIEWS_PATH + "3class-" + balanced + "-test.txt"
        validation_file = self.REVIEWS_PATH + "3class-" + balanced + "-validation.txt"

        open(train_file, 'w').write('\n'.join(map(str, train_ids)))
        open(test_file, 'w').write('\n'.join(map(str, test_ids)))
        open(validation_file, 'w').write('\n'.join(map(str, validation_ids)))
        
        return (train_ids, test_ids)

    # Reads a training or test file. The file contains the indices of the
    # reviews from the clean reviews file.
    def read_file(self, file_name):
        ins = open(file_name).readlines()
        ins = [int(i.strip()) for i in ins]

        return ins

    # A helpter function.
    def set_binary_klass(self, ar):
        ar[(ar == 1) + (ar == 2)] = 0
        ar[(ar == 4) + (ar == 5)] = 1

    # A helpter function.
    def set_ternary_klass(self, ar):
        ar[(ar == 1) + (ar == 2)] = 0
        ar[(ar == 4) + (ar == 5)] = 1
        ar[(ar == 3)] = 2
        
    # Returns (train_x, train_y, test_x, test_y)
    # where x is the review body and y is the rating (1->5 or 0->1)
    def get_train_test(self, klass="2", balanced="balanced"):
        (rating, a, b, c, body) = self.read_clean_reviews()
        rating = np.array(rating)
        body = pd.Series(body)

        train_file = (self.REVIEWS_PATH + klass + "class-" + 
            balanced + "-train.txt")
        test_file = (self.REVIEWS_PATH + klass + "class-" + 
            balanced + "-test.txt")

        train_ids = self.read_file(train_file)
        test_ids = self.read_file(test_file)

        train_y = rating[train_ids]
        test_y = rating[test_ids]
        train_x = body[train_ids]
        test_x = body[test_ids]

        if klass == "2":
            self.set_binary_klass(train_y)
            self.set_binary_klass(test_y)
        return (train_x, train_y, test_x, test_y)

    # Returns (train_x, train_y, test_x, test_y)
    # where x is the review body and y is the rating (1->5 or 0->1)
    def get_train_test_validation(self, klass="3", balanced="balanced"):
        (rating, a, b, c, body) = self.read_clean_reviews()
        rating = np.array(rating)
        body = pd.Series(body)

        train_file = (self.REVIEWS_PATH + klass + "class-" + 
            balanced + "-train.txt")
        test_file = (self.REVIEWS_PATH + klass + "class-" + 
            balanced + "-test.txt")
        validation_file = (self.REVIEWS_PATH + klass + "class-" + 
            balanced + "-validation.txt")
        
        train_ids = self.read_file(train_file)
        test_ids = self.read_file(test_file)
        validation_ids = self.read_file(validation_file)
        
        train_y = rating[train_ids]
        test_y = rating[test_ids]
        valid_y = rating[validation_ids]
        
        train_x = body[train_ids]
        test_x = body[test_ids]
        valid_x = body[validation_ids]
        
        if klass == "2":
            self.set_binary_klass(train_y)
            self.set_binary_klass(test_y)
            self.set_binary_klass(valid_y)
        elif klass == "3":
            self.set_ternary_klass(train_y)
            self.set_ternary_klass(test_y)
            self.set_ternary_klass(valid_y)
        return (train_x, train_y, test_x, test_y, valid_x, valid_y)
    
    
    def split_train_validation_test_3class_tiny(self, rating, tiny_precent, percent_test, percent_valid,
                                balanced="unbalanced"):
        np.random.seed(1234)

        rating = np.array(rating, dtype='int32')
        # length
        num_reviews = len(rating)
        review_ids = np.arange(0, num_reviews)

        # convert to binary, with ratings [1, 2] --> neg  and [4, 5] --> pos
        rating[rating == 2] = 1
        rating[rating == 4] = 5

        ids = (rating == 1) + (rating == 5) + (rating == 3)
        review_ids = review_ids[ids]
        rating = rating[ids]
        rating[rating == 1] = 0
        rating[rating == 5] = 1
        rating[rating == 3] = 2
        # get length after filtering
        num_reviews = rating.shape[0]
        new_data_size = int(np.floor(tiny_precent * rating.shape[0]))
        positive_reviews_precent = np.sum(rating == 1) * 1.0 / rating.shape[0]
        negative_reviews_precent = np.sum(rating == 0) * 1.0 / rating.shape[0]
        neutral_reviews_precent = np.sum(rating == 2) * 1.0 / rating.shape[0]
        new_postive_size = np.round(positive_reviews_precent * tiny_precent * num_reviews) 
        new_negative_size = np.round(negative_reviews_precent * tiny_precent * num_reviews) 
        new_neutral_size = np.round(neutral_reviews_precent * tiny_precent * num_reviews) 
        np.random.shuffle(review_ids)
        selected_ids = np.zeros(new_data_size,dtype='int32')
        i=0
        j=0
        count_pos=0
        count_neg=0
        count_neutral=0
        while(j<new_data_size):
            if(rating[review_ids[i]]==1 and count_pos< new_postive_size):
                selected_ids[j]=np.int(review_ids[i])
                count_pos+=1
                j+=1
            elif(rating[review_ids[i]]==0 and count_neg< new_negative_size):
                selected_ids[j]=np.int(review_ids[i])
                count_neg+=1
                j+=1
            elif(rating[review_ids[i]]==2 and count_neutral< new_neutral_size):
                selected_ids[j]=np.int(review_ids[i])
                count_neutral+=1
                j+=1
            i+=1
            
         
        if balanced == "unbalanced":
            ntest = np.floor(new_data_size * percent_test)
            nvalid = np.floor(new_data_size * percent_valid)
            np.random.shuffle(selected_ids)

            test_ids = selected_ids[:ntest]
            validation_ids = selected_ids[ntest:ntest + nvalid]
            train_ids = selected_ids[ntest + nvalid:]

        elif balanced == "balanced":
            (sizes, bins) = np.histogram(rating, [0, 1, 2, 3])
            min_size = np.min(sizes)
            print min_size

            # sample review ids equally among classes
            test_ids = np.zeros((0,), dtype="int32")
            validation_ids = np.zeros((0,), dtype="int32")
            train_ids = np.zeros((0,), dtype="int32")
            rating = np.array(rating)
            ntest = np.floor(min_size * percent_test)
            nvalid = np.floor(min_size * percent_valid)
            for c in [0, 1, 2]:
                cids = selected_ids[np.nonzero(rating == c)]
                np.random.shuffle(cids)
                
                test_ids = np.r_[test_ids, cids[:ntest]]
                validation_ids = np.r_[validation_ids, cids[ntest:ntest + nvalid]]
                train_ids = np.r_[train_ids, cids[ntest + nvalid:min_size]]

        train_file = self.REVIEWS_PATH + "3class-" + balanced + "-tiny-train.txt"
        test_file = self.REVIEWS_PATH + "3class-" + balanced + "-tiny-test.txt"
        validation_file = self.REVIEWS_PATH + "3class-" + balanced + "-tiny-validation.txt"

        open(train_file, 'w').write('\n'.join(map(str, train_ids)))
        open(test_file, 'w').write('\n'.join(map(str, test_ids)))
        open(validation_file, 'w').write('\n'.join(map(str, validation_ids)))
        
        return (train_ids, test_ids)

# l=LABR()
# (rating, a, b, c, body)=l.read_clean_reviews()
# l.split_train_validation_test_3class_tiny(rating,0.1, 0.2, 0.2)
