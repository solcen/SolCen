# Copyright (c) NUDT.

from tree_sitter import Language, Parser
from DFG.utils import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)

class oDFG(object):
    def __init__(self):
        self.i = 0
    i = 0
    def DFG_solidity(self,root_node, index_to_code, states,node_list):
        assignment = ['assignment_expression', 'augmented_assignment_expression']
        def_statement = ['variable_declaration_statement']
        increment_statement = ['update_expression']
        binary_statement = ['binary_expression']
        if_statement = ['if_statement', 'else']
        for_statement = ['for_statement']
        enhanced_for_statement = []
        while_statement = ['while_statement']
        do_first_statement = []
        function_statement = ["function_definition", "fallback_receive_definition"]
        states = states.copy()
        if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
            #if node_list[self.i+1][3] == 1:
            #    self.i+=1
            #    return [], states
            #self.i+=1
            #print(root_node.text.decode())
            idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
            if root_node.type == code:
                return [], states
            elif code in states:
                return [(code, idx, 'comesFrom', [code], states[code].copy())], states
            else:
                if root_node.type == 'identifier':
                    states[code] = [idx]
                return [(code, idx, 'comesFrom', [], [])], states
        elif root_node.type in function_statement:
            DFG = []
            func_name = root_node.child_by_field_name("function_name")
            if func_name is not None:
                indexs,tag= tree_to_variable_index(func_name, index_to_code)
                #if tag != 0 :
                #    self.i+=tag
                for index in indexs:
                    idx, code = index_to_code[index]
                    DFG.append((code, idx, 'comesFrom', [], []))
                    states[code] = [idx]
            original_states = states.copy()
            for child in root_node.children[2:]:
                temp, states = self.DFG_solidity(child, index_to_code, states,node_list)
                DFG += temp
            return sorted(DFG, key=lambda x: x[1]), original_states
        elif root_node.type in def_statement:
            left_nodes = root_node.child_by_field_name('left')
            right_nodes = root_node.child_by_field_name('right')
            DFG = []
            if not right_nodes is None:
                temp, states = self.DFG_solidity(right_nodes, index_to_code, states,node_list)
                DFG += temp
                name_indexs,tag= tree_to_variable_index(left_nodes, index_to_code)
                value_indexs,tag = tree_to_variable_index(right_nodes, index_to_code)
                for index1 in name_indexs:
                    idx1, code1 = index_to_code[index1]
                    for index2 in value_indexs:
                        idx2, code2 = index_to_code[index2]
                        DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                    states[code1] = [idx1]
                return sorted(DFG, key=lambda x: x[1]), states
            else:
                indexs,tag = tree_to_variable_index(left_nodes, index_to_code)
                for index in indexs:
                    idx, code = index_to_code[index]
                    DFG.append((code, idx, 'comesFrom', [], []))
                    states[code] = [idx]
                return sorted(DFG, key=lambda x: x[1]), states
        elif root_node.type in assignment:
            left_nodes = root_node.child_by_field_name('left')
            right_nodes = root_node.child_by_field_name('right')
            DFG = []
            temp, states = self.DFG_solidity(right_nodes, index_to_code, states,node_list)
            DFG += temp
            name_indexs,tag = tree_to_variable_index(left_nodes, index_to_code)
            value_indexs,tag = tree_to_variable_index(right_nodes, index_to_code)
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(DFG, key=lambda x: x[1]), states
        elif root_node.type in increment_statement:
            DFG = []
            indexs,tag = tree_to_variable_index(root_node, index_to_code)
            for index1 in indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(DFG, key=lambda x: x[1]), states
        #elif root_node.type in binary_statement :
        #    DFG = []
            
        elif root_node.type in if_statement:
            DFG = []
            current_states = states.copy()
            others_states = []
            flag = False
            tag = False
            if 'else' in root_node.type:
                tag = True
            for child in root_node.children:
                if 'else' in child.type:
                    tag = True
                if child.type not in if_statement and flag is False:
                    temp, current_states = self.DFG_solidity(
                        child, index_to_code, current_states,node_list)
                    DFG += temp
                else:
                    flag = True
                    temp, new_states = self.DFG_solidity(child, index_to_code, states,node_list)
                    DFG += temp
                    others_states.append(new_states)
            others_states.append(current_states)
            if tag is False:
                others_states.append(states)
            new_states = {}
            for dic in others_states:
                for key in dic:
                    if key not in new_states:
                        new_states[key] = dic[key].copy()
                    else:
                        new_states[key] += dic[key]
            for key in states:
                if key not in new_states:
                    new_states[key] = states[key]
                else:
                    new_states[key] += states[key]
            for key in new_states:
                new_states[key] = sorted(list(set(new_states[key])))
            return sorted(DFG, key=lambda x: x[1]), new_states
        elif root_node.type in for_statement:
            DFG = []
            for child in root_node.children:
                temp, states = self.DFG_solidity(child, index_to_code, states,node_list)
                DFG += temp
            flag = False
            for child in root_node.children:
                if flag:
                    temp, states = self.DFG_solidity(child, index_to_code, states,node_list)
                    DFG += temp
                elif child.type == "variable_declaration":
                    flag = True
            dic = {}
            for x in DFG:
                if (x[0], x[1], x[2]) not in dic:
                    dic[(x[0], x[1], x[2])] = [x[3], x[4]]
                else:
                    dic[(x[0], x[1], x[2])][0] = list(
                        set(dic[(x[0], x[1], x[2])][0]+x[3]))
                    dic[(x[0], x[1], x[2])][1] = sorted(
                        list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
            DFG = [(x[0], x[1], x[2], y[0], y[1])
                for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
            return sorted(DFG, key=lambda x: x[1]), states
        elif root_node.type in while_statement:
            DFG = []
            for i in range(2):
                for child in root_node.children:
                    temp, states = self.DFG_solidity(child, index_to_code, states,node_list)
                    DFG += temp
            dic = {}
            for x in DFG:
                if (x[0], x[1], x[2]) not in dic:
                    dic[(x[0], x[1], x[2])] = [x[3], x[4]]
                else:
                    dic[(x[0], x[1], x[2])][0] = list(
                        set(dic[(x[0], x[1], x[2])][0]+x[3]))
                    dic[(x[0], x[1], x[2])][1] = sorted(
                        list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
            DFG = [(x[0], x[1], x[2], y[0], y[1])
                for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            DFG = []
            for child in root_node.children:
                if child.type in do_first_statement:
                    temp, states = self.DFG_solidity(child, index_to_code, states,node_list)
                    DFG += temp
            for child in root_node.children:
                if child.type not in do_first_statement:
                    temp, states = self.DFG_solidity(child, index_to_code, states,node_list)
                    DFG += temp

            return sorted(DFG, key=lambda x: x[1]), states
