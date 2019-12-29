from math import log
import copy

class DTNode:
    def __init__(self, x, y, criterion="entropy", parent=None):
        #criterion = entropy | gini
        self.criterion = criterion
        # اطلاعات داخل کلاس
        self.x = x
        # نتیجه کلاس
        self.y = y
        # نود برگ باشد یا نه
        self.isleaf = False
        # پدر دارد یا نه
        self.parent = parent
        # بازه های thresholds
        self.devision_thresholds = []
        self.check = ""
        print(x.columns)
        if(len(x.columns)==0 or len(self.get_y(self.x).unique())==1):
            self.isleaf = True
            self.children = []
            self.target_class = self.get_y(self.x).mode()[0] 
        else:
            self.check = str(len(y.unique()))
            self.children = []
            selected_feature = self.feature_select()
            self.feature = selected_feature
            features = list(self.x.columns)           
            classes = self.classify(self.x ,selected_feature)
            for c in classes:
                dt = c.drop(selected_feature, axis=1)
                self.children.append(DTNode(dt, self.y, self.criterion, parent=self))
    
    def is_leaf(self):
        if self.children == []:
            return True
        else:
            return False
        
    def feature_select(self):
        features = list(self.x.columns)
        l = []
        for f in features:
            l.append(self.information_gain(self.x, f))
            #print("{} = {}".format(f, self.information_gain(self.x, f)))
        l = zip(features, l)
        selected_feature = max(l, key=lambda x:x[1])[0]
        return selected_feature
        
    def entropy(self, y):
        #تعداد کل ستون های جواب
        p_total = y.count()
        #مقدار اولیه e = 0
        e = 0
        # محاسبه انتروپی با فرمول انتروپی
        for p_i in y.value_counts():
            e += (p_i/p_total)*log((p_i/p_total), 2)
        #برگرداندن انتروپی
        return -e

    def giniindex(self, y):
        #تعداد کل ستون های جواب
        p_total = y.count()
        #مقدار اولیه e = 0
        e = 0
        # محاسبه انتروپی با فرمول انتروپی
        for p_i in y.value_counts():
            e += (p_i/p_total)**2
        #برگرداندن انتروپی
        return 1-e
    
    
    def information_gain(self, dataframe, feature):
        #  محاسبه انتروپی کلی 
        entropy_s = self.entropy(self.get_y(dataframe))
        # طول داده 
        s_size = len(dataframe)
        # اجرای تابع classify
        classes = self.classify(dataframe, feature)
        # مقدار اولیه ce
        ce = 0
        for sv in classes:
            # محاسبه طول کلاس
            sv_size = len(sv)
            # محاسبه information gain
            ce+= (sv_size/s_size)*self.entropy(self.get_y(sv))

        igain = entropy_s - ce
        return igain
    
    def evaluate(self, xt):
        if not self.isleaf:
            if(self.feature in xt.index):
                print(type(xt))
                child_index = self.find_child_index(xt[self.feature])
                xt = xt.drop(self.feature)
                return self.children[child_index].evaluate(xt)
            else:
                return self.target_class
        else:
            return self.target_class
        
    
    def classify(self, dataframe, feature):
        # مرتب کردن داده ها
        unique_values = sorted(dataframe[feature].unique())
        #تعریف کردن کلاس خالی
        classes = []
        for v in unique_values:
            # افزودن ویژگی به کلاس
            classes.append(dataframe[dataframe[feature] == v])
        # تعریف thresholds
        self.devision_thresholds = []
        for i, v in enumerate(unique_values[0:-1]):
            self.devision_thresholds.append((unique_values[i]+unique_values[i+1])/2)
        return classes
    
    def find_child_index(self, feature_value):
        for i, v in enumerate(self.devision_thresholds):
            if(feature_value <= v):
                return i
        return len(self.devision_thresholds)
    
    def get_y(self, x):
        return self.y.loc[x.index]

    def prune(self, validation_x, validation_y):
        #reduced error pruning
        mr_without_pruning = self.calc_node_misclassification_rate(validation_x, validation_y)
        # چاپ کردن ارور
        print(mr_without_pruning)
        # یک کوپی از نود برمیدارم
        clone = copy.copy(self)
        # ان را برگ میکنم
        clone.make_it_leaf()
        # برای درخت جدید ارور را محاسبه میکنم
        mr_with_pruning = clone.calc_node_misclassification_rate(validation_x, validation_y)
        print(mr_with_pruning)
        if(mr_without_pruning >= mr_with_pruning):
            self.make_it_leaf()
        for child in self.children:
            child.prune(validation_x, validation_y)

    def make_it_leaf(self):
        #  تبدیل به برگ
        self.target_class = self.get_y(self.x).mode()[0]
        # is leaf == true
        self.isleaf = True
        # فرزندانش رو خالی میکنم
        self.children = []

    def calc_node_misclassification_rate(self, validation_x, validation_y):
        err = 0
        for i in validation_x.index:
            if(self.evaluate(validation_x.loc[i]) != validation_y.loc[i]):
                err +=1
        err = err/len(validation_x)
        return err