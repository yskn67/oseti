oseti
==========
|travis| |coveralls| |pyversion| |version| |license|

Dictionary based Sentiment Analysis for Japanese

INSTALLATION
==============

::

 $ pip install oseti


USAGE
============

.. code:: python

  import oseti

  analyzer = oseti.Analyzer()
  analyzer.analyze('天国で待ってる。')
  # => [1.0]
  analyzer.analyze('遅刻したけど楽しかったし嬉しかった。すごく充実した！')
  # => [0.3333333333333333, 1.0]

ACKNOWLEDGEMENT
=================

This module uses 日本語評価極性辞書（用言編）ver.1.0 and 日本語評価極性辞書（名詞編）ver.1.0

- 小林のぞみ，乾健太郎，松本裕治，立石健二，福島俊一. 意見抽出のための評価表現の収集. 自然言語処理，Vol.12, No.3, pp.203-222, 2005. / Nozomi Kobayashi, Kentaro Inui, Yuji Matsumoto, Kenji Tateishi. Collecting Evaluative Expressions for Opinion Extraction, Journal of Natural Language Processing 12(3), 203-222, 2005.

- 東山昌彦, 乾健太郎, 松本裕治, 述語の選択選好性に着目した名詞評価極性の獲得, 言語処理学会第14回年次大会論文集, pp.584-587, 2008. / Masahiko Higashiyama, Kentaro Inui, Yuji Matsumoto. Learning Sentiment of Nouns from Selectional Preferences of Verbs and Adjectives, Proceedings of the 14th Annual Meeting of the Association for Natural Language Processing, pp.584-587, 2008.


.. |travis| image:: https://travis-ci.org/ikegami-yukino/oseti.svg?branch=master
    :target: https://travis-ci.org/ikegami-yukino/oseti
    :alt: travis-ci.org

.. |coveralls| image:: https://coveralls.io/repos/ikegami-yukino/oseti/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/ikegami-yukino/oseti?branch=master
    :alt: coveralls.io

.. |pyversion| image:: https://img.shields.io/pypi/pyversions/oseti.svg

.. |version| image:: https://img.shields.io/pypi/v/oseti.svg
    :target: http://pypi.python.org/pypi/oseti/
    :alt: latest version

.. |license| image:: https://img.shields.io/pypi/l/oseti.svg
    :target: http://pypi.python.org/pypi/oseti/
    :alt: license
