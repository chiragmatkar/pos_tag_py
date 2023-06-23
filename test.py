def display_results(self, results):
    # Detailed category breakout 
    total_count = 0
    for (category, count) in sorted(results.category_count.items(), key=lambda x: x[1], reverse=True):
      print "%-60s %5s" % (category.full_name(), count)
      print "  " + " ".join(results.category_words[category])
      total_count += count

    # Summary for each top-level category
    top_categories = self.category_tree.children.values()
    def get_top_category(cat):
      for top_cat in top_categories:
        if cat.isa(top_cat):
          return top_cat
      print "Category %s doesn't exist in %s" % (category, top_categories)
        
    top_category_counts = {}
    for top_category in top_categories:
      top_category_counts[top_category] = 0
    
    for category in results.category_count:
      top_category = get_top_category(category)
      if top_category:
        top_category_counts[top_category] += results.category_count[category]

    print ""

    def percent(x, y):
      if y == 0:
        return 0
      else:
        return (100.0 * x) / y

    for top_category in top_categories:
      count = top_category_counts[top_category]
      print "%-20s: %f %%" % (top_category.full_name(), percent(count, total_count))
        
    # Word count
    print "\n%d words total" % (results.word_count,)

  def display_results_html(self, results, title):
    # Detailed category breakout 
    total_count = 0
    print "<html><head>"

    print "<meta http-equiv='content-type' content='text/html; charset=UTF-8'>"
    print """
  <style type="text/css">
    .word-count { vertical-align: super; font-size: 50%; }
    .twisty { color: blue; font-family: monospace; }
    a.twisty { text-decoration: none; }
  </style>
"""
    print "<title>%s</title>" % (title,)
    print """
<script>

var TWISTY_EXPANDED = ' &#9662; ';
var TWISTY_COLLAPSED = ' &#9656; ';

function allWordNodes() {
  var nodes = document.getElementsByTagName("tr");
  var results = new Array();
  var numResults = 0;

  for (i = 0; i < nodes.length; i++) {
    var node = nodes.item(i);
    if (node.className == 'words') {
      results[numResults] = node;
      numResults++;
    }
  }
  return results;
}

function hideAll() {
  allNodes = allWordNodes();
  for (var i = 0; i < allNodes.length; i++) {
    hide(allNodes[i]);
  }
}

function showAll() {
  allNodes = allWordNodes();
  for (var i = 0; i < allNodes.length; i++) {
    show(allNodes[i]);
  }
}

function get_twisty_node(category) {
  var cell = document.getElementById(category + "-cat");
  return cell.childNodes[0];
}

function hide(element) {
  element.style.display = "none";
  var twisty = get_twisty_node(element.id);
  twisty.innerHTML = TWISTY_COLLAPSED;
}

function show(element) {
  element.style.display = "";
  var twisty = get_twisty_node(element.id);
  twisty.innerHTML = TWISTY_EXPANDED;
}

function toggle(cat) {
  var node = document.getElementById(cat)
  if (node.style.display == "none") {
    show(node);
  } else {
    hide(node);
  }
}

</script>
"""
    print "</head><body>"
    print "<h1>%s</h1>" % (title,)
    print "<p><a href='javascript:hideAll()'>- collapse all</a>  <a href='javascript:showAll()'>+ expand all</a></p>"
    print "<table width='100%'>"
    for (category, count) in sorted(results.category_count.items(), key=lambda x: x[1], reverse=True):
      sys.stdout.write("<tr>")
      sys.stdout.write("<td class='%s' id='%s'>" % ("category", category.full_name() + "-cat"))
      sys.stdout.write("""<a class='twisty' href="javascript:toggle('%s')"><span class='twisty'> &#9662; </span></a>""" % (category.full_name(),))
      sys.stdout.write("%s</td><td width='*' align='right'>%s</td></tr>""" % (category.full_name(), count))
      print "<tr class='%s' id='%s'>" % ("words", category.full_name())
      print "<td style='padding-left: 1cm;' colspan='2'>"
      words = uniq_c(results.category_words[category])
      for word in words:
        sys.stdout.write("%s<span class='word-count'>%s</span> " % (word))
      print "\n</td></tr>"
      total_count += count
    print "</table>"
