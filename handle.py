from SolidityLexer import SolidityLexer
from SolidityParser import SolidityParser
import antlr4

class handle(object) :
    def __init__(self, parser):
        self.parser = parser
    def node_del(self,node):
        parent = node.parentCtx
        i = self.child_num(parent,node)
        parent.children[i] = node.children[0]
        node.children[0].parentCtx = parent 
        del node
    def child_num(self,parent,child):
        for i in range(0,parent.getChildCount()):
            if parent.children[i] is child :
                return i
    def walk_tree(self,tree,lev,type) :
        tree,lev = self.operate(tree,type,lev)
        if hasattr(tree,'symbol'):
            return
        
        
        for i in range(0,tree.getChildCount()) :
            if hasattr(tree.children[i],'symbol') == False :
                if self.parser.ruleNames[tree.children[i].getRuleIndex()] == 'primaryExpression':
                    x = 1
            #self.operate(tree.children[i],type,lev+1)
            self.walk_tree(tree.children[i],lev+1,type)
    def operate(self,node,type,lev):
        #Remove antlr4 keywords in tree which have only one child 
        if type == 0:
            if (node.getChildCount() == 1) & (hasattr(node,'symbol') == False) :
                parent = node.parentCtx
                self.node_del(node)
                return parent,lev-1
            else:
                return node,lev
        if type == 1:
            if hasattr(node,'symbol'):
                print(' '*lev+'| '+node.symbol.text)
            else :
                print(' '*lev+'| '+self.parser.ruleNames[node.getRuleIndex()])
            return node,lev