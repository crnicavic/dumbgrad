from dumbgrad.utils import *
from dumbgrad.nn import *
from sklearn.model_selection import train_test_split
import csv

"""
     1. cap-shape:                bell=b,conical=c,convex=x,flat=f,
                                  knobbed=k,sunken=s
     2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
     3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,
                                  pink=p,purple=u,red=e,white=w,yellow=y
     4. bruises?:                 bruises=t,no=f
     5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,
                                  musty=m,none=n,pungent=p,spicy=s
     6. gill-attachment:          attached=a,descending=d,free=f,notched=n
     7. gill-spacing:             close=c,crowded=w,distant=d
     8. gill-size:                broad=b,narrow=n
     9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,
                                  green=r,orange=o,pink=p,purple=u,red=e,
                                  white=w,yellow=y
    10. stalk-shape:              enlarging=e,tapering=t
    11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
                                  rhizomorphs=z,rooted=r,missing=?
    12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
    13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
    14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    16. veil-type:                partial=p,universal=u
    17. veil-color:               brown=n,orange=o,white=w,yellow=y
    18. ring-number:              none=n,one=o,two=t
    19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
                                  none=n,pendant=p,sheathing=s,zone=z
    20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
                                  orange=o,purple=u,white=w,yellow=y
    21. population:               abundant=a,clustered=c,numerous=n,
                                  scattered=s,several=v,solitary=y
    22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
                                  urban=u,waste=w,woods=d

Since the features of this dataset are strings and letters
the idea is to turn this into a one hot encoded dataset.

The intended way to do that is to go through all the columns,
and turn every value into the format: <column_name>_<feature_value>
Then all of those become the columns themselves, and the values are just
0 and 1.

The idea is to form a row like (shortened for clarity):
class, cap-shape, cap-surface
EDIBLE,KNOBBED,SMOOTH
POISONOUS,KNOBBED,SCALY

turn this into:
class_edible, class_poisonous, cap_shape_knobbed, cap_surface_smooth, cap_surface_scaly
1,0,1,1,0
0,1,1,0,1
"""

if __name__ == "__main__":
    shrooms_file = open("./examples/datasets/mushroom.csv", mode='r')
    reader = csv.reader(shrooms_file)
    shrooms = list(reader)
    header, shrooms = shrooms[0], shrooms[1:]
    categorized = {}
    for col, col_name in zip(list(zip(*shrooms)), header):
        col = list(map(lambda c: 'UNKNOWN' if c == '?' else c, col))
        uniq = unique(col)
        for k in uniq:
            categorized[f"{col_name}_{k}"] = [int(k == c) for c in col]

    y_cols = ["class_EDIBLE", "class_POISONOUS"]
    x_cols = list(filter(lambda c: c not in y_cols, list(categorized.keys())))
    y = [categorized[c] for c in y_cols]
    y = [list(row) for row in list(zip(*y))]

    x = [categorized[c] for c in x_cols]
    x = [list(row) for row in list(zip(*x))]

    num_classes = 2
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=0)
    n = Network([
        Input(len(x[0])),
        Layer(30),
        Layer(30),
        Layer(num_classes, activation="softmax")
    ])
    opt = Optimizer(lr=0.01)
    reg = L2Regularization()
    n.build(seed=0, loss="cross_entropy", regularization=reg, optimizer=opt)
    n.train(x_train, y_train, epochs=10, batch_size=43, n_jobs=4)
    n.test(x_test, y_test)
