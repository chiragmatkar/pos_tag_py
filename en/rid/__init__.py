#!/usr/bin/env python
#
# Copyright 2007 John Wiseman <jjwiseman@yahoo.com>
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import sys
from io import StringIO 
import getopt
from . import rid_dict

class RegressiveImageryDictionary:
  """
  To use:
    1. Load a dictionary.
    2. Load an exclusion list (optional).
    3. Call analyze.
    4. Call display_results with the value returned by analyze.    
  """
  def __init__(self):
    self.category_tree = CategoryRoot()
    self.exclusion_patterns = []
    self.exclusion_pattern = None
    self.pattern_tree = DiscriminationTree('root', None)

  def load_dictionary_from_file(self, path):
    rid_in = open(path, "r")
    try:
      self.load_dictionary(rid_in)
    finally:
      rid_in.close()

  def load_dictionary_from_string(self, string):
    rid_in = StringIO(string)
    self.load_dictionary(rid_in)
      
  def load_dictionary(self, stream):
    primary_category = None
    secondary_category = None
    tertiary_category = None

    for line in stream:
      num_tabs = count_leading_tabs(line)
      # The dictionary is in kind of a weird format.
      if num_tabs == 0:
        primary_category = line.strip()
        secondary_category = None
        tertiary_category = None
      elif num_tabs == 1:
        secondary_category = line.strip()
        tertiary_category = None
      elif num_tabs == 2 and not '(' in line:
        tertiary_category = line.strip()
      else:
        # We have a word pattern.
        pattern = line.strip().split(' ')[0].lower()
        category = self.ensure_category(primary_category, secondary_category, tertiary_category)
        category.add_word_pattern(pattern)
        self.pattern_tree.put(pattern, category)

  def load_exclusion_list_from_file(self, path):
    exc_in = open(path, "r")
    try:
      self.load_exclusion_list(exc_in)
    finally:
      exc_in.close()
      
  def load_exclusion_list_from_string(self, string):
    exc_in = StringIO(string)
    self.load_exclusion_list(exc_in)

  def load_exclusion_list(self, stream):
    for line in stream:
      pattern = line.strip().lower()
      pattern = pattern.replace("*", ".*")
      self.exclusion_patterns.append(pattern)
    # One megapattern to exclude them all
    self.exclusion_pattern = re.compile('^(' + '|'.join(self.exclusion_patterns) + ')$')

  def token_is_excluded(self, token):
    return self.exclusion_pattern.match(token)

  def get_category(self, word):
    categories = self.pattern_tree.retrieve(word)
    if categories:
      return categories[0]
          
  def analyze(self, text):
    results = RIDResults()
    def increment_category(category, token):
      if not category in results.category_count:
        results.category_count[category] = 0
        results.category_words[category] = []
      results.category_count[category] += 1
      results.category_words[category].append(token)
    
    tokens = tokenize(text)
    results.word_count = len(tokens)
    for token in tokens:
      if not self.token_is_excluded(token):
        category = self.get_category(token)
        if category != None:
          increment_category(category, token)
    return results

  def display_results(self, results):
    # Detailed category breakout 
    total_count = 0
    for (category, count) in sorted(results.category_count.items(), key=lambda x: x[1], reverse=True):
      print("%-60s %5s") % (category.full_name(), count)
      print("  " + " ".join(results.category_words[category]))
      total_count += count

    # Summary for each top-level category
    top_categories = self.category_tree.children.values()
    def get_top_category(cat):
      for top_cat in top_categories:
        if cat.isa(top_cat):
          return top_cat
      print("Category %s doesn't exist in %s") % (category, top_categories)
        
    top_category_counts = {}
    for top_category in top_categories:
      top_category_counts[top_category] = 0
    
    for category in results.category_count:
      top_category = get_top_category(category)
      if top_category:
        top_category_counts[top_category] += results.category_count[category]

    print("")

    def percent(x, y):
      if y == 0:
        return 0
      else:
        return (100.0 * x) / y

    for top_category in top_categories:
      count = top_category_counts[top_category]
      print("%-20s: %f %%") % (top_category.full_name(), percent(count, total_count))
        
    # Word count
    print("\n%d words total") % (results.word_count,)

#   def display_results(self, results):
#     # Detailed category breakout 
#     total_count = 0
#     for (category, count) in sorted(results.category_count.items(), key=lambda x: x[1], reverse=True):
#       print("%-60s %5s")% (category.full_name(), count)
#       print("  " + " ".join(results.category_words[category]))
#       total_count += count

#     # Summary for each top-level category
#     top_categories = self.category_tree.children.values()
#     def get_top_category(cat):
#       for top_cat in top_categories:
#         if cat.isa(top_cat):
#           return top_cat
#       print("Category %s doesn't exist in %s") % (category, top_categories)
        
#     top_category_counts = {}
#     for top_category in top_categories:
#       top_category_counts[top_category] = 0
    
#     for category in results.category_count:
#       top_category = get_top_category(category)
#       if top_category:
#         top_category_counts[top_category] += results.category_count[category]

#     print("")

#     def percent(x, y):
#       if y == 0:
#         return 0
#       else:
#         return (100.0 * x) / y

#     for top_category in top_categories:
#       count = top_category_counts[top_category]
#       print("%-20s: %f %%" % (top_category.full_name(), percent(count, total_count)))
        
#     # Word count
#     print("\n%d words total" % (results.word_count,))

#   def display_results_html(self, results, title):
#     # Detailed category breakout 
#     total_count = 0
    
	
#     html="""
#      <html><head>
#    <meta http-equiv='content-type' content='text/html; charset=UTF-8'>
# 	<style type="text/css">
#     .word-count { vertical-align: super; font-size: 50%; }
#     .twisty { color: blue; font-family: monospace; }
#     a.twisty { text-decoration: none; }
#   </style>
# <script>
# var TWISTY_EXPANDED = ' &#9662; ';
# var TWISTY_COLLAPSED = ' &#9656; ';

# function allWordNodes() {
#   var nodes = document.getElementsByTagName("tr");
#   var results = new Array();
#   var numResults = 0;

#   for (i = 0; i < nodes.length; i++) {
#     var node = nodes.item(i);
#     if (node.className == 'words') {
#       results[numResults] = node;
#       numResults++;
#     }
#   }
#   return results;
# }

# function hideAll() {
#   allNodes = allWordNodes();
#   for (var i = 0; i < allNodes.length; i++) {
#     hide(allNodes[i]);
#   }
# }

# function showAll() {
#   allNodes = allWordNodes();
#   for (var i = 0; i < allNodes.length; i++) {
#     show(allNodes[i]);
#   }
# }

# function get_twisty_node(category) {
#   var cell = document.getElementById(category + "-cat");
#   return cell.childNodes[0];
# }

# function hide(element) {
#   element.style.display = "none";
#   var twisty = get_twisty_node(element.id);
#   twisty.innerHTML = TWISTY_COLLAPSED;
# }

# function show(element) {
#   element.style.display = "";
#   var twisty = get_twisty_node(element.id);
#   twisty.innerHTML = TWISTY_EXPANDED;
# }

# function toggle(cat) {
#   var node = document.getElementById(cat)
#   if (node.style.display == "none") {
#     show(node);
#   } else {
#     hide(node);
#   }
# }

# </script>
#   <title>{title}</title>
#   </head><body>
#   <h1>{title}</h1>
#   <p><a href='javascript:hideAll()'>- collapse all</a>  <a href='javascript:showAll()'>+ expand all</a></p>
#    """
   
	




#     print "<table width='100%'>"
#     for (category, count) in sorted(results.category_count.items(), key=lambda x: x[1], reverse=True):
#       sys.stdout.write("<tr>")
#       sys.stdout.write("<td class='%s' id='%s'>" % ("category", category.full_name() + "-cat"))
#       sys.stdout.write("""<a class='twisty' href="javascript:toggle('%s')"><span class='twisty'> &#9662; </span></a>""" % (category.full_name(),))
#       sys.stdout.write("%s</td><td width='*' align='right'>%s</td></tr>""" % (category.full_name(), count))
#       print "<tr class='%s' id='%s'>" % ("words", category.full_name())
#       print "<td style='padding-left: 1cm;' colspan='2'>"
#       words = uniq_c(results.category_words[category])
#       for word in words:
#         sys.stdout.write("%s<span class='word-count'>%s</span> " % (word))
#       print "\n</td></tr>"
#       total_count += count
#     print "</table>"


    # Summary for each top-level category
    top_categories = self.category_tree.children.values()
    def get_top_category(cat):
      for top_cat in top_categories:
        if cat.isa(top_cat):
          return top_cat
      print("Category %s doesn't exist in %s") % (category, top_categories)
        
    top_category_counts = {}
    for top_category in top_categories:
      top_category_counts[top_category] = 0
    
    for category in results.category_count:
      top_category = get_top_category(category)
      if top_category:
        top_category_counts[top_category] += results.category_count[category]


    def percent(x, y):
      if y == 0:
        return 0
      else:
        return (100.0 * x) / y

    print("<table>")
    for top_category in top_categories:
      count = top_category_counts[top_category]
      print("<tr><td>%s:</td><td>%f %%</td></tr>") % (top_category.full_name(), percent(count, total_count))
    print("<table>")
        
    # Word count
    print("<p>%d words total</p>") % (results.word_count,)
    print("</body></html>")


  def ensure_category(self, *args):
    def ensure_cat_aux(category, category_path):
      if len(category_path) == 0 or category_path[0] == None:
        return category
      else:
        cat = category_path.pop(0)
        if not cat in category.children:
          category.children[cat] = Category(cat, category)
        return ensure_cat_aux(category.children[cat], category_path)
    return ensure_cat_aux(self.category_tree, list(args))


class RIDResults:
  def __init__(self):
    self.category_count = {}
    self.category_words = {}
    self.word_count = 0


WORD_REGEX = re.compile(r'[^a-zA-Z]+')
def tokenize(string):
  tokens = WORD_REGEX.split(string.lower())
  tokens = filter(lambda token: len(token) > 0, tokens)
  return tokens


def count_leading_tabs(string):
  for i, char in enumerate(string):
    if char != '\t':
      return i


class DiscriminationTree:
  """
  This is the discrimination tree we use for mapping words to
  categories.  The put method is used to insert category nodes in the
  tree, associated with some word pattern.  The retrieve method finds
  the category for a given word, if one exists.
  """
  def __init__(self, index, parent):
    self.index = index
    self.parent = parent
    self.leaves = []
    self.interiors = []
    self.is_wildcard = False

  def __str__(self):
    return "<DiscriminationTree %s>" % (self.index,)
 
  def child_matching_index(self, index):
    for child in self.interiors:
      if child.index == index:
        return child
    return None

  def retrieve(self, path):
    if len(path) == 0 or self.is_wildcard:
      return self.leaves
    else:
      next_index = path[0]
      next_disc_tree = self.child_matching_index(next_index)
      if next_disc_tree == None:
        return
      else:
        return next_disc_tree.retrieve(path[1:])

  def put(self, path, leaf):
    if len(path) == 0:
      if isinstance(leaf, DiscriminationTree):
        self.interiors.append(leaf)
      else:
        self.leaves.append(leaf)
      return True
    else:
      next_index = path[0]
      if next_index == '*':
        # Got a '*' so this is a wildcard node that will match
        # anything that reaches it.
        self.is_wildcard = True
        self.leaves.append(leaf)
      else:
        next_disc_tree = self.child_matching_index(next_index)
        if next_disc_tree == None:
          next_disc_tree = DiscriminationTree(next_index, self)
          self.interiors.append(next_disc_tree)
        next_disc_tree.put(path[1:], leaf)

  def dump(self, stream=sys.stdout, indent=0):
    stream.write("\n" + " "*indent + str(self))
    for child in self.leaves:
      stream.write("\n" + " "*(indent + 3) + str(child))
    for child in self.interiors:
      child.dump(stream=stream, indent=indent + 3)


class Category:
  def __init__(self, name, parent):
    self.name = name
    self.parent = parent
    self.children = {}
    self.leaves = []

  def __str__(self):
    return "<Category %s>" % (self.full_name(),)

  def add_word_pattern(self, pattern):
    self.leaves.append(pattern)

  def full_name(self):
    if self.parent == None or isinstance(self.parent, CategoryRoot):
      return self.name
    else:
      return self.parent.full_name() + ":" + self.name

  def isa(self, parent):
    return parent == self or (self.parent and self.parent.isa(parent))


class CategoryRoot(Category):
  def __init__(self):
    Category.__init__(self, 'root', None)

  def full_name(self):
    return ""


def uniq_c(words):
  words.sort()
  results = []
  last_word = words[0]
  last_word_count = 1
  for word in words[1:]:
    if word == last_word:
      last_word_count += 1
    else:
      results.append((last_word, last_word_count))
      last_word = word
      last_word_count = 1
  results.append((last_word, last_word_count))
  results = sorted(results, key=lambda x: x[1], reverse=True)
  return results
  


class RIDApp:
  def usage(self, args):
    print("usage: %s [-h [-t TITLE] | -d FILE | -e FILE | --add-dict=FILE | --add-exc=FILE]") % (args[0],)
    print("%s reads from standard input and writes to standard output.") % (args[0],)
    print("options:")
    print("    -h                Generate HTML output.")
    print("    -t TITLE          Use TITLE as the report heading.")
    print("    -d FILE           Replaces the built-in dictionary with FILE.")
    print("    -e FILE           Replaces the built-in exclusion list with FILE.")
    print("    --add-dict=FILE   Processes FILE as a category dictionary.")
    print("    --add-exc=FILE    Processes FILE as an exlusion list.")
    
  def run(self, args):
    rid = RegressiveImageryDictionary()
    load_default_dict = True
    load_default_exc = True
    html_output = False
    title = "RID Analysis"

    try:
      optlist, args = getopt.getopt(sys.argv[1:], 'd:e:ht:',
                                    ['add-dict=', 'add-exc='])
      for (o, v) in optlist:
        if o == '-d':
          rid.load_dictionary_from_file(v)
          load_default_dict = False
        elif o == '-e':
          rid.load_exclusion_list_from_file(v)
          load_default_exc = False
        elif o == '--add-dict':
          rid.load_dictionary_from_file(v)
        elif o == '--add-exc':
          rid.load_exclusion_list_from_file(v)
        elif o == '-h':
          html_output = True
        elif o == '-t':
          title = v
        else:
          sys.stderr.write("%s: illegal option '%s'\n" % (args[0], o))
          self.usage(args)
    except getopt.GetoptError as e:
      sys.stderr.write("%s: %s\n" % (args[0], e.msg))
      self.usage(args)
      sys.exit(1)
                       
    if load_default_dict:
      rid.load_dictionary_from_string(rid_dict.DEFAULT_RID_DICTIONARY)
    if load_default_exc:
      rid.load_exclusion_list_from_string(rid_dict.DEFAULT_RID_EXCLUSION_LIST)
      
    results = rid.analyze(sys.stdin.read())
    if html_output:
      rid.display_results_html(results, title)
    else:
      rid.display_results(results)
      
if __name__ == '__main__':
  app = RIDApp()
  app.run(sys.argv)

#######################################################################################################

# From trac.util.compat.py
# Implementation for sorted() for Python versions prior to 2.4

try:
	reversed = reversed
except NameError:
	def reversed(x):
		if hasattr(x, 'keys'):
			raise ValueError('mappings do not support reverse iteration')
		i = len(x)
		while i > 0:
			i -= 1
			yield x[i]

try:
	sorted = sorted
except NameError:
	def sorted(iterable, cmp=None, key=None, reverse=False):
		"""Partial implementation of the "sorted" function from Python 2.4"""
		if key is None:
			lst = list(iterable)
		else:
			lst = [(key(val), idx, val) for idx, val in enumerate(iterable)]
			lst.sort()
			if key is None:
				if reverse:
					return lst[::-1]
				return lst
			if reverse:
				lst = reversed(lst)
			return [i[-1] for i in lst]

#######################################################################################################

rid = RegressiveImageryDictionary()
rid.load_dictionary_from_string(rid_dict.DEFAULT_RID_DICTIONARY)
rid.load_exclusion_list_from_string(rid_dict.DEFAULT_RID_EXCLUSION_LIST)

# A list of subcategories for each top category, e.g. emotions ->
# ['anxiety', 'glory', 'positive affect', 'sadness', 'expressive behavior', 'affection', 'aggression']
primary   = [key.lower() for key in rid.category_tree.children["PRIMARY"].children.keys()]
secondary = [key.lower() for key in rid.category_tree.children["SECONDARY"].children.keys()]
emotions  = [key.lower() for key in rid.category_tree.children["EMOTIONS"].children.keys()]

class RIDScoreItem:
	
	def __init__(self, name, count, words, type):
		
		self.name  = name
		self.count = count
		self.words = words
		self.type  = type
		
	def __str__(self):
		
		return self.name

class RIDScore(list):
	
	def __init__(self, rid, results):
	
		self.primary = 0
		self.secondary = 0
		self.emotions = 0

		self.count(rid, results)
		self.populate(results)

	def count(self, rid, results):

		# Keep a count per top category
		# (primary, secondary, emotional)
		score = {}
		roots = rid.category_tree.children
		for key in roots:
			score[key] = 0

		# Calculate the total count.
		# Increase the count for the top category each category belongs to.
		total = 0
		for category in results.category_count:
			count = results.category_count[category]
			total += count
			for key in roots:
				if category.isa(roots[key]):
					score[key] += count

		# Relativise the score for each top category.
		if total > 0:
			for key in score:
				score[key] = float(score[key]) / total

		self.primary = score["PRIMARY"]
		self.secondary = score["SECONDARY"]
		self.emotions = score["EMOTIONS"]

	def populate(self, results):
		
		# A RIDScore is a sorted list of category items,
		# with relevant words found in the text assigned to each category.
		for (category, count) in sorted(results.category_count.items(), key=lambda x: x[1], reverse=True):
			c = RIDScoreItem(
				name=category.name.lower(),
				count=count,
				words=results.category_words[category],
				type=category.parent.name.lower()
			)
			self.append(c)
			
	def __str__(self):
		return str([str(item) for item in self])

def categorise(txt):

	global rid
	results = rid.analyze(txt)
	return RIDScore(rid, results)