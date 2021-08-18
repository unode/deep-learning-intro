---
layout: lesson
root: .  # Is the only page that doesn't follow the pattern /:path/index.html
permalink: index.html  # Is the only page that doesn't follow the pattern /:path/index.html
---

{% include_relative _meta/description.md %}

<!-- this is an html comment -->

{% comment %} This is a comment in Liquid {% endcomment %}

<div class="prereq" markdown="1">
## Prerequisites
{% include_relative _meta/prerequisites.md %}
</div>

{% include links.md %}
