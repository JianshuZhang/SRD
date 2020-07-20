import numpy

import cPickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

def dataIterator(feature_file,mask_file,label_file,align_file,realign_file,dictionary,
                redictionary,batch_size,max_xlen,max_ylen):
    
    fp_feature=open(feature_file,'r')
    features=pkl.load(fp_feature)
    fp_feature.close()

    fp_mask=open(mask_file,'r')
    masks=pkl.load(fp_mask)
    fp_mask.close()

    fp_label=open(label_file,'r')
    labels=pkl.load(fp_label)
    fp_label.close()

    fp_align=open(align_file,'r')
    aligns=pkl.load(fp_align)
    fp_align.close()

    fp_realign=open(realign_file,'r')
    realigns=pkl.load(fp_realign)
    fp_realign.close()

    targets = {}
    retargets = {}
    # map word to int with dictionary
    for uid, label_lines in labels.iteritems():
        char_list = []
        relation_list = []
        for line_idx, line in enumerate(label_lines):
            parts = line.strip().split('\t')
            char = parts[0]
            relation = parts[2]
            if dictionary.has_key(char):
                char_list.append(dictionary[char])
            else:
                print 'a symbol not in the dictionary !! formula',uid ,'symbol', char
                sys.exit()

            if line_idx != 0 and line_idx != len(label_lines)-1:
                if redictionary.has_key(relation):
                    relation_list.append(redictionary[relation])
                else:
                    print 'a relation not in the redictionary !! formula',uid ,'relation', relation
                    sys.exit()
            else:
                relation_list.append(0) # whatever which one
        targets[uid]=char_list
        retargets[uid]=relation_list

    featureLen={}
    for uid,fea in features.iteritems():
        featureLen[uid]=len(fea)

    featureLen= sorted(featureLen.iteritems(), key=lambda d:d[1]) # sorted by sentence length,  return a list with each triple element

    feature_batch=[]
    mask_batch=[]
    label_batch=[]
    relabel_batch=[]
    alignment_batch=[]
    realignment_batch=[]

    feature_total=[]
    mask_total=[]
    label_total=[]
    relabel_total=[]
    alignment_total=[]
    realignment_total=[]

    uidList=[]

    i=0
    for uid,length in featureLen:
        fea=features[uid]
        mask=masks[uid]
        lab=targets[uid]
        relab=retargets[uid]
        ali=aligns[uid]
        reali=realigns[uid]
        if len(lab)>max_ylen:
            print 'this latex length bigger than', max_ylen, 'ignore'
        elif len(fea)>max_xlen:
            print 'this formula length bigger than', max_xlen, 'ignore'
        else:
            uidList.append(uid)
            if i==batch_size: # a batch is full
                feature_total.append(feature_batch)
                mask_total.append(mask_batch)
                label_total.append(label_batch)
                relabel_total.append(relabel_batch)
                alignment_total.append(alignment_batch)
                realignment_total.append(realignment_batch)

                i=0
                feature_batch=[]
                mask_batch=[]
                label_batch=[]
                relabel_batch=[]
                alignment_batch=[]
                realignment_batch=[]
                feature_batch.append(fea)
                mask_batch.append(mask)
                label_batch.append(lab)
                relabel_batch.append(relab)
                alignment_batch.append(ali)
                realignment_batch.append(reali)
                i=i+1
            else:
                feature_batch.append(fea)
                mask_batch.append(mask)
                label_batch.append(lab)
                relabel_batch.append(relab)
                alignment_batch.append(ali)
                realignment_batch.append(reali)
                i=i+1

    # last batch
    feature_total.append(feature_batch)
    mask_total.append(mask_batch)
    label_total.append(label_batch)
    relabel_total.append(relabel_batch)
    alignment_total.append(alignment_batch)
    realignment_total.append(realignment_batch)

    print 'total ',len(feature_total), 'batch data loaded'

    return zip(feature_total,mask_total,label_total,relabel_total,alignment_total,realignment_total), uidList



def dataIterator_valid(feature_file,label_file,dictionary,batch_size,max_xlen,max_ylen):
    
    fp=open(feature_file,'rb') # read kaldi scp file
    features=pkl.load(fp) # load features in dict
    fp.close()

    fp2=open(label_file,'r')
    labels=fp2.readlines()
    fp2.close()

    targets={}
    # map word to int with dictionary
    for l in labels:
        tmp=l.strip().split()
        uid=tmp[0]
        w_list=[]
        for w in tmp[1:]:
            if dictionary.has_key(w):
                w_list.append(dictionary[w])
            else:
                print 'a phone not in the dictionary !! sentence ',uid,'phone ', w
                sys.exit()
        targets[uid]=w_list

    sentLen={}
    for uid,fea in features.iteritems():
        sentLen[uid]=len(fea)

    sentLen= sorted(sentLen.iteritems(), key=lambda d:d[1]) # sorted by sentence length,  return a list with each triple element


    feature_batch=[]
    label_batch=[]
    feature_total=[]
    label_total=[]
    uidList=[]

    i=0
    for uid,length in sentLen:
        fea=features[uid]
        # cmvn
        #fea=(fea-fea_mean)/fea_std


        lab=targets[uid]
        if len(lab)>max_ylen:
            print 'this latex length bigger than', max_ylen, 'ignore'
        elif len(fea)>max_xlen:
            print 'this formula length bigger than', max_xlen, 'ignore'
        else:
            uidList.append(uid)
            if i==batch_size: # a batch is full
                feature_total.append(feature_batch)
                label_total.append(label_batch)

                i=0
                feature_batch=[]
                label_batch=[]
                feature_batch.append(fea)
                label_batch.append(lab)
                i=i+1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i=i+1

    # last batch
    feature_total.append(feature_batch)
    label_total.append(label_batch)

    print 'total ',len(feature_total), 'batch data loaded'

    return zip(feature_total,label_total),uidList
