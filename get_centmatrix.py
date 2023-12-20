import copy
import tempfile
from time import time
import json
import threading
import networkx as nx
import antlr4
from DFG.detect import convert_to_dfg
from SolidityLexer import SolidityLexer
from SolidityParser import SolidityParser
import enum
from anytree import AnyNode
import csv
from multiprocessing import Lock, Pool, Manager

from handle import handle

dict_path = './dict.json'
wrong_path = './wrong.csv'
key_word = ['call','msg','now', 'timestamp', 'require', 'assert']
key_word_not = ['call','msg','now','require', 'assert']
attr_word = ['timestamp','sender','value']
tp_key_word = ['timestamp','now']  # key words directly connected to the timestamp reliability 
ruleNames = ["sourceUnit", "pragmaDirective", "pragmaName", "pragmaValue",
             "version", "versionOperator", "versionConstraint", "importDeclaration",
             "importDirective", "contractDefinition", "inheritanceSpecifier",
             "contractPart", "stateVariableDeclaration", "fileLevelConstant",
             "usingForDeclaration", "structDefinition", "modifierDefinition",
             "modifierInvocation", "functionDefinition", "functionDescriptor",
             "returnParameters", "modifierList", "eventDefinition",
             "enumValue", "enumDefinition", "parameterList", "parameter",
             "eventParameterList", "eventParameter", "functionTypeParameterList",
             "functionTypeParameter", "variableDeclaration", "typeName",
             "userDefinedTypeName", "mappingKey", "mapping", "functionTypeName",
             "storageLocation", "stateMutability", "block", "statement",
             "expressionStatement", "ifStatement", "tryStatement",
             "catchClause", "whileStatement", "simpleStatement", "uncheckedStatement",
             "forStatement", "inlineAssemblyStatement", "doWhileStatement",
             "continueStatement", "breakStatement", "returnStatement",
             "throwStatement", "emitStatement", "variableDeclarationStatement",
             "variableDeclarationList", "identifierList", "elementaryTypeName",
             "expression", "primaryExpression", "expressionList",
             "nameValueList", "nameValue", "functionCallArguments",
             "functionCall", "assemblyBlock", "assemblyItem", "assemblyExpression",
             "assemblyMember", "assemblyCall", "assemblyLocalDefinition",
             "assemblyAssignment", "assemblyIdentifierOrList", "assemblyIdentifierList",
             "assemblyStackAssignment", "labelDefinition", "assemblySwitch",
             "assemblyCase", "assemblyFunctionDefinition", "assemblyFunctionReturns",
             "assemblyFor", "assemblyIf", "assemblyLiteral", "subAssembly",
             "tupleExpression", "typeNameExpression", "numberLiteral",
             "identifier", "hexLiteral", "overrideSpecifier", "stringLiteral"]#antlr4 rule names
load_dict = {}
dict_len = 0
base_vector = None
str_cent = ['cent_harm', 'cent_eigen', 'cent_close', 'cent_between', 'cent_degree'] # centrality types
grey = []
def get_keys(d, value):
    return [k for k,v in d.items() if v == value]
class centrality(object):
    def __init__(self, pid,fun,path):
        self.nodes_tmp = []
        self.place = 0
        self.preorder = 0
        self.edges_tmp = []
        self.pid = pid
        self.code_tokens = []
        self.dfg = []
        self.fun = fun
        self.grey_tokens = []
        self.tag = 0
        self.path = path
        self.pre_word = {}
    code_tokens = []
    grey_tokens = []
    pre_word = {}
    dfg = []
    place = 0
    fun = ""
    tag = 0
    nodes_tmp = []
    edges_tmp = []
    weight = []
    def add_nodes(self):
        for k, v in load_dict.items():
            self.nodes_tmp.append(v)
        for i in range(len(ruleNames)):
            self.nodes_tmp.append(i + dict_len + 1)
    
    def create_anytree(self, trg, src):
        # handle leaf noeds
        if hasattr(src, 'symbol'):
            text = src.symbol.text.strip()
            if text in load_dict:
                trg.index = load_dict[text]
                trg.label = text
                trg.addr = self.place
                self.pre_word[trg.addr] = trg.label
                self.place+=1
                trg.order = self.preorder
                self.preorder+=1
                self.nodes_tmp.append((trg.index,trg.order))
            else:
                trg.index = -1
                trg.label = text
                trg.addr = self.place
                self.pre_word[trg.addr] = trg.label
                self.place+=1
                trg.order = self.preorder
                self.preorder+=1
        # handle antlr4 rule nodes
        else:
            trg.index = load_dict[ruleNames[src.getRuleIndex()]]
            trg.label = ruleNames[src.getRuleIndex()]
            trg.order = self.preorder
            self.preorder+=1
            self.nodes_tmp.append((trg.index,trg.order))
            for child in src.getChildren():
                newnode = AnyNode(index=0, label="", addr=-1, order=0,parent=trg)
                self.create_anytree(newnode, child)
    #analyze DFG to get grey words
    def analyze(self,type):
        list_grey = []
        if type == "tp" :
            for edge in self.dfg:
                if len(edge[3]) != 0:
                    for i in edge[4] :
                        if (edge[3][0] in tp_key_word) or ((edge[3][0],i)in list_grey):
                            list_grey.append((edge[0],edge[1]))
                else :
                    if edge[0] in tp_key_word :
                        list_grey.append((edge[0],edge[1]))
            return list_grey
    def create_graph(self, tree):
        g = nx.Graph()
        g.add_nodes_from(self.nodes_tmp)
        try :
            self.code_tokens,self.dfg=convert_to_dfg(self.fun)
        except Exception as e:
            print(self.path)
            print(e)
            
        list_grey = self.analyze('tp')
        
        self.get_edge(tree,g,list_grey)
        return g,list_grey
    # add special weight to edge included grey words
    def get_edge(self, parent,G,grey_word):
        for i in range(0,len(parent.children)):
            weight = 1
            c_word = (parent.children[i].label,parent.children[i].addr)
            if (c_word in grey_word):
                weight = self.place-8
                G.add_edge((parent.index,parent.order), (parent.children[i].index,parent.children[i].order), weight = weight )
                for j in range(0,len(parent.children)):
                    if j == i :
                        continue
                    else :
                        weight = weight - abs(j-i)
                        G.add_edge((parent.index,parent.order), (parent.children[j].index,parent.children[j].order), weight = weight )
            else :
                G.add_edge((parent.index,parent.order), (parent.children[i].index,parent.children[i].order), weight = weight )
            self.get_edge(parent.children[i],G,grey_word)

    def middle(self, tree, file_id,ifspec):
        # build the tree in anytree format
        newtree = AnyNode(index=0, label="", addr=0)
        self.create_anytree(newtree, tree)
        g,list_grey= self.create_graph(newtree)
        # cnetrality part
        if len(list_grey) != 0:
            tag = 1
        else :
            tag = 0
        if ifspec == 1:
            tag = 0
        cent_matrix = {"file_id": file_id,
                       "isgrey": tag,
                       "cent_harm": list(nx.harmonic_centrality(g,distance='weight').values()),
                       "cent_eigen": list(nx.eigenvector_centrality(g, max_iter=600,weight='weight').values()),
                       "cent_close": list(nx.closeness_centrality(g,distance='weight').values()),
                       "cent_between": list(nx.betweenness_centrality(g,weight='weight').values()),
                       "cent_degree": list(nx.degree_centrality(g).values())}

        return cent_matrix


varlist = {"identifier"}
numlist = {"numberLiteral", "hexLiteral"}
strlist = {"stringLiteral"}
calllist = {"functionCallArguments"}
symbol_token = enum.Enum(
    'symbol_token', ('var', 'num', 'str', 'call', 'array'))

class token(object):
    def __init__(self, pid):
        self.list_array = []
        self.list_attr = []
        self.list_call = []
        self.list_num = []
        self.list_str = []
        self.list_var = []
        self.list_safe = []
        self.list_of = []
        self.num_attr = 0
        self.flag = 0
        self.flag_of = 0
        self.flag_require = 0
        self.pid = pid
    flag_require = 0
    flag_of = 0
    num_attr = 0
    list_var = []
    list_num = []
    list_str = []
    list_call = []
    list_array = []
    list_attr = []
    flag = 0
    list_of = []
    list_safe = []
    # Tokenize function name
    def func_desc_token(self,node,parser):
        if hasattr(node, 'symbol'):
            node.symbol.text = 'function'
        else:
            if node == None:
                return
            for child in node.getChildren():
                self.func_desc_token(child,parser)
                
            if node.getChildCount() == 1:
                func_name = copy.copy(node.children[0])
                func_name.symbol.text = 'function'
                node.addChild(func_name)
    # Traverse parameters and tokenize them
    def param_token(self,node,parser):
        if hasattr(node, 'symbol'):
            return
        if hasattr(node.getChild(0), 'symbol'):
            if parser.ruleNames[node.getRuleIndex()] not in varlist:
                return
            if node.getChild(0).symbol.text in self.list_var:
                node.getChild(0).symbol.text = "var" + str(self.list_var.index(node.getChild(0).symbol.text))
                # self.num_var+=1
            else:
                self.list_var.append(node.getChild(0).symbol.text)
                node.getChild(0).symbol.text = "var" + str(self.list_var.index(node.getChild(0).symbol.text))
        else:
            for child in node.getChildren():
                self.param_token(child, parser)
    # Traverse contents and tokenize all leaf nodes
    def expr_token(self, expr, parser, iftype, expr_lev):
        self.flag = 0
        if expr.getChildCount() == 0:
            return None
        if parser.ruleNames[expr.getRuleIndex()] == "variableDeclarationStatement":
            self.num_attr = 0
            for child in expr.getChildren():
                self.expr_token(child, parser, iftype, 1)
        if parser.ruleNames[expr.getRuleIndex()] == "variableDeclaration":
            self.varDec_token(expr, parser)
        if parser.ruleNames[expr.getRuleIndex()] == "expression":
            if expr.getChildCount() == 4:
                if hasattr(expr.getChild(1), 'symbol') and hasattr(expr.getChild(3), 'symbol'):
                    if expr.getChild(1).symbol.text == "[" and expr.getChild(3).symbol.text == "]":
                        if self.flag == 0 :
                            if self.flag_of == 1 :
                                self.list_of.append(expr.getRuleContext().getText())
                            else :
                                self.list_safe.append(expr.getRuleContext().getText())
                        self.flag = 1

            for expr_child in expr.getChildren():
                if hasattr(expr_child, 'symbol'):
                    if expr_child.symbol.text == ".":
                        self.num_attr = 1
                        continue
                if expr_child.getChildCount() == 0:
                    continue
                if self.num_attr == 1:
                    self.num_attr = 0
                    if expr_child.getChild(0).symbol.text in key_word :
                        continue
                    if expr_child.getChild(0).symbol.text in attr_word :
                        continue
                    if expr_child.getChild(0).symbol.text in self.list_attr:
                        expr_child.getChild(0).symbol.text = "attr" + str(
                            self.list_attr.index(expr_child.getChild(0).symbol.text))
                    else:
                        self.list_attr.append(expr_child.getChild(0).symbol.text)
                        expr_child.getChild(0).symbol.text = "attr" + str(
                            self.list_attr.index(expr_child.getChild(0).symbol.text))
                    continue
                if parser.ruleNames[expr_child.getRuleIndex()] == "expression":
                    if self.flag:
                        self.expr_token(expr_child, parser, symbol_token.array.value, expr_lev + 1)
                        self.flag = 0
                    else:
                        self.expr_token(expr_child, parser, iftype, expr_lev + 1)
                if parser.ruleNames[expr_child.getRuleIndex()] in calllist:
                    self.expr_token(expr.getChild(0), parser, symbol_token.call.value, expr_lev)
                    self.expr_token(expr_child, parser, iftype, expr_lev + 1)

                if parser.ruleNames[expr_child.getRuleIndex()] == "primaryExpression":
                    a = expr_child
                    if hasattr(expr_child.getChild(0), 'symbol'):
                        continue
                    if parser.ruleNames[expr_child.getChild(0).getRuleIndex()] in numlist:
                        if expr_child.getChild(0).getChild(0) is not None:
                            if expr_child.getChild(0).getChild(0).symbol.text in key_word_not :
                                continue
                            if expr_child.getChild(0).getChild(0).symbol.text in self.list_num:
                                expr_child.getChild(0).getChild(0).symbol.text = "num" + str(
                                    self.list_num.index(expr_child.getChild(0).getChild(0).symbol.text))
                            else:
                                self.list_num.append(expr_child.getChild(0).getChild(0).symbol.text)
                                expr_child.getChild(0).getChild(0).symbol.text = "num" + str(
                                    self.list_num.index(expr_child.getChild(0).getChild(0).symbol.text))

                    elif parser.ruleNames[expr_child.getChild(0).getRuleIndex()] in strlist:
                        if expr_child.getChild(0).getChild(0) is not None:
                            if expr_child.getChild(0).getChild(0).symbol.text in key_word_not :
                                continue
                            if expr_child.getChild(0).getChild(0).symbol.text in self.list_str:
                                expr_child.getChild(0).getChild(0).symbol.text = "str" + str(
                                    self.list_str.index(expr_child.getChild(0).getChild(0).symbol.text))
                            else:
                                self.list_str.append(expr_child.getChild(0).getChild(0).symbol.text)
                                expr_child.getChild(0).getChild(0).symbol.text = "str" + str(
                                    self.list_str.index(expr_child.getChild(0).getChild(0).symbol.text))

                    elif iftype == symbol_token.call.value:
                        if expr_child.getChild(0).getChild(0) is not None:
                            if hasattr(expr_child.getChild(0).getChild(0), 'symbol') is False:
                                elechild = expr_child.getChild(0)
                                elechild.getChild(0).getChild(0).symbol.text = "element"
                                continue
                            if expr_child.getChild(0).getChild(0).symbol.text in key_word_not :
                                continue
                            if expr_child.getChild(0).getChild(0).symbol.text in self.list_call:
                                expr_child.getChild(0).getChild(0).symbol.text = "func" + str(
                                    self.list_call.index(expr_child.getChild(0).getChild(0).symbol.text))
                            else:
                                self.list_call.append(expr_child.getChild(0).getChild(0).symbol.text)
                                expr_child.getChild(0).getChild(0).symbol.text = "func" + str(
                                    self.list_call.index(expr_child.getChild(0).getChild(0).symbol.text))

                    elif iftype == symbol_token.var.value:
                        if expr_child.getChild(0).getChild(0) is not None:
                            if hasattr(expr_child.getChild(0).getChild(0), 'symbol') is False:
                                elechild = expr_child.getChild(0)
                                elechild.getChild(0).getChild(0).symbol.text = "element"
                                continue
                            if expr_child.getChild(0).getChild(0).symbol.text in key_word_not :
                                continue
                            if expr_child.getChild(0).getChild(0).symbol.text in self.list_var:
                                expr_child.getChild(0).getChild(0).symbol.text = "var" + str(
                                    self.list_var.index(expr_child.getChild(0).getChild(0).symbol.text))
                            else:
                                self.list_var.append(expr_child.getChild(0).getChild(0).symbol.text)
                                expr_child.getChild(0).getChild(0).symbol.text = "var" + str(
                                    self.list_var.index(expr_child.getChild(0).getChild(0).symbol.text))

                    elif iftype == symbol_token.array.value:
                        if expr_child.getChild(0).getChild(0) is not None:
                            if hasattr(expr_child.getChild(0).getChild(0), 'symbol') is False:
                                elechild = expr_child.getChild(0)
                                elechild.getChild(0).getChild(0).symbol.text = "element"
                                continue
                            if expr_child.getChild(0).getChild(0).symbol.text in key_word_not :
                                continue
                            if expr_child.getChild(0).getChild(0).symbol.text in self.list_array:
                                expr_child.getChild(0).getChild(0).symbol.text = "array" + str(
                                    self.list_array.index(expr_child.getChild(0).getChild(0).symbol.text))
                            else:
                                self.list_array.append(expr_child.getChild(0).getChild(0).symbol.text)
                                expr_child.getChild(0).getChild(0).symbol.text = "array" + str(
                                    self.list_array.index(expr_child.getChild(0).getChild(0).symbol.text))



        else:
            for i in range(0, expr.getChildCount()):
                child = expr.getChild(i)
                self.expr_token(child, parser, iftype, expr_lev)

    def addspace(self, node):
        if hasattr(node, 'symbol'):
            node.symbol.text = ' ' + node.symbol.text + ' '
        else:
            for child in node.getChildren():
                self.addspace(child)
    # Tokenize the variable definition statements
    def varDec_token(self, node, parser):
        if hasattr(node, 'symbol'):
            return
        if hasattr(node.getChild(0), 'symbol'):
            if parser.ruleNames[node.getRuleIndex()] not in varlist:
                return
            if node.getChild(0).symbol.text in self.list_var:
                node.getChild(0).symbol.text = "var" + str(self.list_var.index(node.getChild(0).symbol.text))

            else:
                self.list_var.append(node.getChild(0).symbol.text)
                node.getChild(0).symbol.text = "var" + str(self.list_var.index(node.getChild(0).symbol.text))
        else:
            for child in node.getChildren():
                self.varDec_token(child, parser)


def string_to_file(string):
    file_like_obj = tempfile.NamedTemporaryFile()
    file_like_obj.write(string.encode())

    file_like_obj.flush()
    file_like_obj.seek(0)
    return file_like_obj

def get_single_cent_matrix(context, labels, label, all_num, folder, type):
    try:
        pid = threading.get_ident()
        fi = string_to_file(context['contract content'])
        input_ = antlr4.FileStream(fi.name, encoding='utf-8')
        lexer = SolidityLexer(input_)
        tokens = antlr4.CommonTokenStream(lexer)
        parser = SolidityParser(tokens)
        parser._errHandler = antlr4.BailErrorStrategy()
        tree = parser.sourceUnit()
        fi.close()
    except:
        lock.acquire()
        all_num.value += 1
        print("file %d finish" % all_num.value)
        lock.release()
        res = []
        with open(wrong_path, 'a') as f:
            writer = csv.writer(f)
            res.append(context['contract name'])
            res.append("wrong")
            writer.writerow(res)
        print("%s has wrong!" % context['contract name'])
        return

    change = token(pid)

    ifspec = 0
    for part_child in tree.functionDefinition():
        i = 0
        func_desc = part_child.functionDescriptor()
        change.func_desc_token(func_desc,parser)
        param_list = part_child.parameterList()
        if hasattr(param_list,'parameter'):
            for param in param_list.parameter():
                change.param_token(param,parser)
        if hasattr(part_child,'returnParameters'):
            return_param = part_child.returnParameters()
            if hasattr(return_param,'parameterList'):
                param_list = return_param.parameterList()
                if hasattr(param_list,'parameter'):
                    for param in param_list.parameter():
                        change.param_token(param,parser)
        if hasattr(part_child,'modifierList'):
            modifier = part_child.modifierList().getText()
            if (modifier.find('private') != -1) :
                ifspec = 1
                break

        if isinstance(part_child.block(), list):
            for func_child in part_child.block():
                if hasattr(func_child, 'statement'):
                    for block_child in func_child.statement():
                        change.expr_token(block_child, parser, symbol_token.var.value, 0)
        else:
            func_child = part_child.block()
            if hasattr(func_child, 'statement'):
                for block_child in func_child.statement():
                    change.expr_token(block_child, parser, symbol_token.var.value, 0)
    ex_name = folder.split('.')[0]
    change.addspace(tree)
    sub = handle(parser)
    sub.walk_tree(tree,0,0)
    array = tree.getRuleContext().getText()
    analyze = centrality(pid,array,context['contract name'])
    file_id = ex_name+'_'+context['contract name']
    result= analyze.middle(tree, file_id,ifspec)
    
    lock.acquire()
    all_num.value += 1
    print(ex_name + " file %d finish" % all_num.value)

    if result is not None:
        if type == 1:
            filename = './test1.json'
            with open(filename, 'a') as f:
                json.dump(result, f)
                f.write('\n')
            filename = './'+ex_name + '.json'
            with open(filename, 'a') as f:
                json.dump(result, f)
                f.write('\n')
        else:
            filename = './test2.json'
            with open(filename, 'a') as f:
                json.dump(result, f)
                f.write('\n')
            filename = './'+ex_name + '.json'
            with open(filename, 'a') as f:
                json.dump(result, f)
                f.write('\n')
        # labels.append([file_id, label])
    lock.release()

# init process pool
def init_pool(lo):
    global lock
    lock = lo


def main(path_name, type):
    global dict_len,load_dict
    lock = Lock()
    # preprocess
    lock.acquire()
    with open(dict_path, 'r') as f:
        load_dict = json.load(f)
        dict_len = len(load_dict)
        for i in range(len(key_word)):
            load_dict[key_word[i]] = dict_len+i
        dict_len = len(load_dict)
        for i in range(len(ruleNames)):
            load_dict[ruleNames[i]] = i + dict_len + 2
        dict_len = len(load_dict)
    lock.release()
    labels = Manager().list()
    all_num = Manager().Value('i', 0)
    

    # multiprocess
    pool = Pool(processes=48, initializer=init_pool, initargs=(lock,), maxtasksperchild=1)
    with open(path_name,'r',encoding = "utf8") as reader:
        while True:
            text = reader.readline()
            if text:
                text_line = json.loads(text)
                pool.apply_async(get_single_cent_matrix, (text_line, labels, 1, all_num, path_name, type))
            else:
                break
    pool.close()
    pool.join()
    return pool



