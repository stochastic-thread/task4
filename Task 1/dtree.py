__author__ = "arthur"

import pandas as pd

pd.set_option('display.max_rows', 1000)

def indices_next(tlist):
	count = 0
	elt = -100
	indices = []
	for t in tlist:
		if (elt < t[1]):
			indices.append(count)
			elt = t[1]
		count += 1
	return indices

def summed_list(ls):
	for elt in range(0, len(ls)):
		if elt != 0:
			ls[elt] += ls[elt-1]
	return ls

class TreeNode(object):
	class_counter = 0
	def __init__(self):
		self.name = TreeNode.class_counter
		TreeNode.class_counter += 1
		self.split_gini = -1000
		self.data = pd.DataFrame()
		self.node_type = "Node"
		self.node_gini = 1.0
		self.split_value = -1000
		self.split_attribute = ""
		self.parent = None
		self.left_child_node = None
		self.left_child_complete = False
		self.split_dict = dict()

		self.right_child_node = None
		self.right_child_complete = False
		self.level = 0

	def compute_gini_new_node(self):
		split_dict = self.data["OK"].value_counts().to_dict()
		self.split_dict = split_dict
		if len(split_dict) == 2:
			# print "Size of split_dict is 2"
			zero_count = float(split_dict[0])
			one_count = float(split_dict[1])
			gini = 1 - (zero_count/(zero_count+one_count))**2 - (one_count/(zero_count+one_count))**2
			self.node_gini = gini

class DecisionTree(object):
	def __init__(self):
		self.root = TreeNode()
		self.attributes = []
		self.used_attributes = set()

	def is_leaf_node(self, node):
		result = False
		data_ct = len(node.data)
		if len(node.split_dict) == 1:
			result = True
		elif len(node.split_dict) == 2:
			zeroes = node.split_dict[0]
			ones = node.split_dict[1]
			if zeroes == 0 or ones == 0:
				result = True
			else:
				result = False
			if zeroes == 0 and ones == 0:
				result = False
		else:
			result = False
		return result

	def create(self, filename):
		current_node = self.root
		current_node.data = pd.read_csv(filename)
		self.attributes = current_node.data.columns

	def create_test(self, filename):
		csv = pd.read_csv(filename)
		current_node = self.root
		current_node.data = csv[:int((0.8*len(csv)))]
		self.attributes = current_node.data.columns

		testing_data = csv[int((0.8*len(csv))):]

		self.train_tree()

		for i in range(400, 400+len(testing_data)):
			current_node = self.root
			record = testing_data.loc[i]
			while current_node.left_child_node is not None and current_node.right_child_node is not None:
				attr = current_node.split_attribute
				if current_node.split_value <= record[attr]:
					current_node = current_node.right_child_node
					continue
				else:
					current_node = current_node.left_child_node
					continue
			if len(current_node.split_dict) == 2:
				zero = current_node.split_dict[0]
				#print current_node.split_dict
				one = current_node.split_dict[1]
				zeropc = zero/(float(zero + one))

				onepc = one/(float(zero + one))
				if max(zeropc, onepc) == zeropc:
					print str(record['ID']) + " is a 0 " + str(zeropc)
				else:
					print str(record['ID']) + " is a 1 " + str(onepc)


	def train_tree(self):
		self.train_tree_hidden(self.root)

	def get_attribute_ginis(self, current_node):
		attribute_ginis = dict()
		hold_ginis = []
		for attribute in self.attributes[1:10]:
			if attribute not in self.used_attributes:
				# print self.used_attributes
				attribute_ginis[attribute] = []
				# hold_ginis will hold the gini coefficients of every possible splitting condition to find the best one
				# attribute_df uses built in pandas functions to sort by attribute, THEN by ID
				attribute_df = current_node.data.sort([attribute,"ID"])

				# attribute_vals are the actual sorted values of the individual attribute
				attribute_vals = attribute_df[attribute]

				# buckets is a histogram of the different attribute value counts.
				buckets = attribute_vals.value_counts()

				#print attribute

				# since attribute_vals is sorted, we can use this to know the offset
				series = buckets.sort_index()

				# series_keys = series.keys().tolist()
				# for key in series_keys:
				#     offset = series[key]
				#     print str(key) + " " + str(offset)

				summedlist = summed_list(series.tolist())
				#print summedlist
				#return
				count = 0
				for element in summedlist:

					# we get the sorted list of attribute values, and using elt, the summed indices, we grab the
					# data that's been sectioned up (see attribute_vals, buckets, etc)
					subsection = attribute_df[:element]
					# last_val = subsection[-1:]

					# this is a series
					val_counts = subsection["OK"].value_counts()
					series_size = val_counts.size
					if series_size == 2:
						# then we know this node will split and it is not a leaf node
						left = val_counts[0]
						right = val_counts[1]
						if left != 0 or right != 0:
							gini = 1 - (float(left)/(left+right))**2 - (float(right)/(left+right))**2

							# hold_ginis.append(tpl)
					elif series_size == 1:
						if len(subsection) > 0:
							gini = 0
					tpl = (gini, count, element, attribute)
					attribute_ginis[attribute].append(tpl)
					count += 1
		return attribute_ginis


	def split(self, current_node):
		attributes_start = current_node.data.columns[1:10]

		attribute_ginis = self.get_attribute_ginis(current_node)

		# print attribute_ginis
		tuple_dict = dict()
		slimmer_ginis = []
		for attribute in attributes_start:
			tuple_dict[attribute] = []
			if attribute not in self.used_attributes:
				# print attribute_ginis
				tuple_list = sorted(attribute_ginis[attribute],  key=lambda x: x[0])
				if len(tuple_list) > 0:
					slimmer_ginis.append(tuple_list[0])
					# No, the first tuple is not necessarily the best.
					# first_tuple_is_best = sorted(slimmer_ginis, key= lambda x: x[0])
				for tpl in tuple_list:
					if tpl[2] != 500:
						left_df = current_node.data.sort([attribute,"ID"])[:tpl[2]]
						right_df = current_node.data.sort([attribute,"ID"])[tpl[2]:]
						#print "first half"
						#left_freq = left_df[attribute].value_counts()
						#print "second half"

						# right_freq = right_df[attribute].value_counts()

						left_freq = left_df["OK"].value_counts()

						right_freq = right_df["OK"].value_counts()
						leftsum = 0
						rightsum = 0
						if len(left_freq) == 0:
							left_gini = 0
						elif len(left_freq) == 1:
							left_gini = 0
						else:

							left_gini = 1 - (left_freq[0]/float(left_freq[0] + left_freq[1]))**2 - (left_freq[1]/float(left_freq[0] + left_freq[1]))**2
							leftsum = left_freq[0] + left_freq[1]

						#print "LG " + str(left_gini)

						if len(right_freq) == 0:
							right_gini = 0
						elif len(right_freq) == 1:
							right_gini = 0
						else:
							right_gini = 1 - (right_freq[0]/float(right_freq[0] + right_freq[1]))**2 - (right_freq[1]/float(right_freq[0] + right_freq[1]))**2
							rightsum = right_freq[0] + right_freq[1]

						#print "RG " + str(right_gini)
						if left_gini == 0 or right_gini == 0:
							continue
						else:

							split_gini = left_gini*(leftsum/float(leftsum+rightsum)) + right_gini*(rightsum/float(leftsum+rightsum))
							#print "SG " + str(split_gini)
							info_gain = current_node.node_gini - split_gini
							#print info_gain

							tuple_dict[attribute].append((tpl[0], tpl[1], tpl[2], attribute, left_gini, right_gini, split_gini, info_gain))
						# print tuple_dict
		total_list = []
		#print tuple_dict
		for key in tuple_dict.keys():
			for val in tuple_dict[key]:
				total_list.append(val)
		list = sorted(total_list,  key=lambda x: x[7])

		if len(list) == 0:
			return

		x = list[0]

		gini = x[0]
		valsplit = x[1]
		ind = x[2]

		attribute_to_split_on = x
		# smallest_gini_value_tuple = start_tuple[0]
		current_node.node_gini = gini
		best_attribute_to_split_on_based_on_gini = x[3]
		best_attribute = best_attribute_to_split_on_based_on_gini
		attribute_df = current_node.data.sort([best_attribute,"ID"])
		# print len(attribute_df)


		current_node.split_value = x[1]
		# attribute_df[:ind][-1:][best_attribute].tolist()[0]


		current_node.split_attribute = best_attribute

		current_node.left_child_node = TreeNode()
		current_node.left_child_node.parent = current_node

		current_node.right_child_node = TreeNode()
		current_node.right_child_node.parent = current_node

		current_node.left_child_node.data = attribute_df[:ind]
		current_node.right_child_node.data = attribute_df[ind:]

		# print best_attribute
		# print "left node data count " + str(len(current_node.left_child_node.data))
		# print "LEFT NODE DATA BEGINS"
		# print "--------------"
		# #print current_node.left_child_node.data
		# print "--------------"
		# print "LEFT NODE DATA ENDS"

		lnode_data = current_node.left_child_node.data
		#
		# print "right node data count " + str(len(current_node.right_child_node.data))
		# print "RIGHT NODE DATA BEGINS"
		# print "--------------"
		# #print current_node.right_child_node.data
		# print "--------------"
		# print "RIGHT NODE DATA ENDS"
		rnode_data = current_node.right_child_node.data

		current_node.left_child_node.compute_gini_new_node()
		current_node.right_child_node.compute_gini_new_node()

		left_half_split_gini = (float(len(lnode_data))/len(current_node.data))*current_node.left_child_node.node_gini
		right_half_split_gini = (float(len(rnode_data))/len(current_node.data))*current_node.right_child_node.node_gini

		current_node.split_gini = left_half_split_gini + right_half_split_gini

		# print "Split gini " + str(current_node.split_gini)
		# print current_node.split_attribute
		# print "I am " + str(current_node.name)
		# if current_node.parent is not None:
		# 	print "My parent is " + str(current_node.parent.name)
		# print "LTE " + str(current_node.split_value)
		# print "left is " + str(current_node.left_child_node.data["OK"].value_counts().to_dict())
		# print "right is " + str(current_node.right_child_node.data["OK"].value_counts().to_dict())
		#
		#
		# left_gini = current_node.left_child_node.node_gini
		# right_gini = current_node.right_child_node.node_gini
		#
		# print "left gini is " + str(left_gini)
		#
		# print "right gini is " + str(right_gini)
		# print ""

		self.used_attributes.add(x[3])

	def train_tree_hidden(self, current_node):
		not_done = True

		while not_done:
			self.split(current_node)

			if current_node.left_child_node is None and current_node.right_child_node is None:
				print "Current node is a leaf node!!!"
				return
			else:
				left_gini = current_node.left_child_node.node_gini
				right_gini = current_node.right_child_node.node_gini
				split_gini = current_node.split_gini
				# print "do i get here"
				left_gain = (left_gini - split_gini)
				right_gain = (right_gini - split_gini)

				if max(left_gain, right_gain) == left_gain:
					current_node = current_node.left_child_node
				else:
					current_node = current_node.right_child_node
				continue



	def test(self, filename):
		current_node = self.root

		csv = pd.read_csv(filename)
		freq = dict()
		freq[0] = 0
		freq[1] = 0
		print "ID,OK"
		for i in range(0, len(csv)):
			current_node = self.root
			record = csv.loc[i]
			while current_node.left_child_node is not None and current_node.right_child_node is not None:
				attr = current_node.split_attribute
				if current_node.split_value <= record[attr]:
					current_node = current_node.right_child_node
					continue
				else:
					current_node = current_node.left_child_node
					continue
			if len(current_node.split_dict) == 2:
				zero = current_node.split_dict[0]
				# print current_node.split_dict
				one = current_node.split_dict[1]
				zeropc = zero/(float(zero + one))

				onepc = one/(float(zero + one))
				if max(zeropc, onepc) == zeropc:
					print str(record['ID'])+",0"
					freq[0] += 1
				else:
					print str(record['ID']) + ",1" # + str(onepc)
					freq[1] += 1
		#print freq






		# print csv

def main(filename):
	filename = "training.csv"

	dt = DecisionTree()
	dt.create(filename)
	dt.train_tree()
	dt.test("test.csv")

	# dt.create_test(filename)




# dt.test("test.csv")

main("training.csv")