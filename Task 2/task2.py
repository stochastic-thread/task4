import pandas as pd
import itertools 
pd.set_option('display.max_rows', 10000)
class DataSet:
    def __init__(self,filename):
        self.df = pd.read_csv(filename)
        self.headers = self.df.columns
        self.attributes = self.headers[1:11]
        self.attribute_values = dict()
        for x in self.attributes:
            self_df = self.df[x]
            vc = self_df.value_counts(sort=False)
            self.attribute_values[x] = vc

d = DataSet("training.csv")

every_combo = []
for attribute in d.attributes:
    attr_values = d.attribute_values[attribute].keys()
    for attr_value in attr_values:
        for eq in ["<","==",">"]:
            every_combo.append((attribute, attr_value, eq))

attr_hash = {}            
csv = pd.read_csv("training.csv")
matrix = []
m4 = []
for x in every_combo:
    rl = []
    (attr,val,eq) = x
    if attr not in attr_hash.keys():
        attr_hash[attr] = []
    else:
        valx = attr_hash[attr]
        newdf = csv
        if eq == "<":
            newdf = csv[csv[attr] < val]
        elif eq == "==":
            newdf = csv[csv[attr] == val]
        elif eq == ">":
            newdf = csv[csv[attr] > val]

        zo_dict = dict(newdf["OK"].value_counts())
        crit = eq
        if len(zo_dict) == 2:
            zero_count = zo_dict[0]
            one_count = zo_dict[1]
            if max(zero_count, one_count) == zero_count:
                rl = [attr, crit, val, sum(zo_dict.values()), zo_dict, (float(zero_count)/(zero_count+one_count))]
            else:
                rl = [attr, crit, val, sum(zo_dict.values()), zo_dict, (float(one_count)/(zero_count+one_count))]
        elif len(zo_dict) == 1:
            count = None
            if list(zo_dict.keys())[0] == 0:
                zero_count = zo_dict[0]
                count = zero_count
            else:
                one_count = zo_dict[1]
                count = one_count
            if count == one_count:
                rl = [attr, crit, val, sum(zo_dict.values()), zo_dict, 1.0]
            else:
                rl = [attr, crit, val, sum(zo_dict.values()), zo_dict, 1.0]
    m4.append(rl)

mxx = pd.DataFrame(m4,columns=["attr", "crit", "val", "sz", "ok_counts", "confidence"])
#m1 = mtrx[(mtrx['sz'] > 100) & (mtrx['sz'] != 500) & (mtrx['attr'] != "OK")]
mx2 = mxx[(mxx['sz'] > 100) & (mxx['sz'] != 500) & (mxx['attr'] != "OK")]
mx3 = mx2[mx2["confidence"] >= 0.8]
mx3

headers = csv.columns[1:10]
attr_hash = {}
for h in headers:
    attr_hash[h] = mx3[mx3['attr'] == h]

list_hash = {}
for k in attr_hash.keys():
    attr_df = attr_hash[k]
    list_hash[k] = []
    for (i, row) in attr_df.iterrows():
        a = row['attr']
        c = row['crit']
        v = row['val']
        d = row['ok_counts']
        cf = row['confidence']
        list_hash[k].append((a, c, v))

listX = []
lists = []
for k in list_hash:
    lists.append(list_hash[k])

for l in lists: 
    l.insert(0, None)

for x in itertools.product(*lists):
    if len(list(filter(None, x))) < 4:
        listX.append(list(filter(None, x)))

print("hi")
csvX = pd.read_csv("training.csv")
mtr5 = []
for elt in listX:
    csv = csvX
    for x in elt:
        attr = x[0]
        eq = x[1]
        val = x[2]
        if eq == "<":
            csv = csv[csv[attr] < val]
        elif eq == "==":
            csv = csv[csv[attr] == val]
        elif eq == ">":
            csv = csv[csv[attr] > val]
    zo_dict = dict(csv["OK"].value_counts())
    rl = []
    if len(zo_dict) == 2:
        zero_count = zo_dict[0]
        one_count = zo_dict[1]
        if max(zero_count, one_count) == zero_count:
            rl = [elt, sum(zo_dict.values()), zo_dict, 0, (float(zero_count)/(zero_count+one_count))]
        else:
            rl = [elt, sum(zo_dict.values()), zo_dict, 1, (float(one_count)/(zero_count+one_count))]
    elif len(zo_dict) == 1:
        count = None
        if list(zo_dict.keys())[0] == 0:
            zero_count = zo_dict[0]
            count = zero_count
        else:
            one_count = zo_dict[1]
            count = one_count
        if count == one_count:
            rl = [elt, sum(zo_dict.values()), zo_dict, 1, 1.0]
        else:
            rl = [elt, sum(zo_dict.values()), zo_dict, 0, 1.0]

    if len(elt) > 1:
        mtr5.append(rl)    
mt5 = pd.DataFrame(mtr5)


hello = mt5.sort([1],ascending=False)
bye = hello[(hello[4] > 0.8) & (hello[1] > 100)]
zero = bye[bye[3] == 0]
one = bye[bye[3] == 1]
oo = one.sort(ascending=False)
zz = zero.sort(ascending=False)

for (index, row) in testing.iterrows():
    record = {}
    for attr in testing.columns[1:10]:
        record[attr] = row[attr]
    
    count = 0
    for (rind,rowbye) in zz.iterrows():
        for ind, constraint in enumerate(rowbye[0]):
            # print(constraint)
            
            at = constraint[0]
            cr = constraint[1]
            vl = constraint[2]
            
            if cr == "<":
                if record[at] < vl:
                    count += 1
                    pass
            elif cr == "==":
                if record[at] == vl:
                    count += 1
                    pass
            elif cr == ">":
                if record[at] > vl:
                    count += 1
                    pass
        if (count == len(rowbye[0])):
            print(str(row["ID"]) +","+ str(int(rowbye[3])))
            break
        else:
            print(str(row["ID"]) +",1")
            break