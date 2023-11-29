{% if referencefile %}
.. include:: {{ referencefile }}
{% endif %}

{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:
   :show-inheritance:

{% block attributes_summary %}
   {% if attributes %}
   .. rubric:: Attributes
      {% for item in all_attributes %}
         {%- if not item.startswith('_') %}
   .. autoattribute:: {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
   {% endif %}
{% endblock %}

{% block methods_summary %}
   {% if methods %}
   .. rubric:: Methods
   {% for item in all_methods %}
      {%- if not item.startswith('_') or item in ['__call__', '__mul__', '__getitem__', '__len__'] %}
   .. automethod:: {{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}

   {% endif %}
{% endblock %}