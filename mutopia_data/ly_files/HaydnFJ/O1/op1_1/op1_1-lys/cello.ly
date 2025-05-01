\version "2.18.2"
\language "deutsch"

\paper {
  ragged-bottom = ##t
  ragged-last-bottom = ##t
}

\header {
  title = "String Quartet in B-flat major"
  subtitle = \markup { \abs-fontsize #14 "Violoncello" }
  composer = "Joseph Haydn (1732-1809)"
  opus = "Op. 1 No. 1"

  mutopiacomposer = "HaydnFJ"
  mutopiainstrument = "String Quartet"
  date = "ca. 1757-62"
  source = "Trautwein & Comp, Berlin"
  style = "Classical"
  license = "Creative Commons Attribution 4.0"
  maintainer = "Urs Metzger"

 footer = "Mutopia-2018/02/08-2213"
 copyright = \markup {\override #'(font-name . "DejaVu Sans, Bold") \override #'(baseline-skip . 0) \right-column {\with-url #"http://www.MutopiaProject.org" {\abs-fontsize #9  "Mutopia " \concat {\abs-fontsize #12 \with-color #white \char ##x01C0 \abs-fontsize #9 "Project "}}}\override #'(font-name . "DejaVu Sans, Bold") \override #'(baseline-skip . 0 ) \center-column {\abs-fontsize #11.9 \with-color #grey \bold {\char ##x01C0 \char ##x01C0 }}\override #'(font-name . "DejaVu Sans,sans-serif") \override #'(baseline-skip . 0) \column { \abs-fontsize #8 \concat {"Typeset using " \with-url #"http://www.lilypond.org" "LilyPond " \char ##x00A9 " 2018 " "by " \maintainer " " \char ##x2014 " " \footer}\concat {\concat {\abs-fontsize #8 { \with-url #"http://creativecommons.org/licenses/by/4.0/" "Creative Commons Attribution 4.0 International License "\char ##x2014 " free to distribute, modify, and perform" }}\abs-fontsize #13 \with-color #white \char ##x01C0 }}}
 tagline = ##f
}

\include "defs.ily"
\include "vc_a.ily"
\include "vc_b.ily"
\include "vc_c.ily"
\include "vc_d.ily"
\include "vc_e.ily"

\score {
  \Cello_a
  \header {
    piece = \markup { \hspace #53.2 \abs-fontsize #16 \bold \raise #3 { "I." }}
  }
  \layout { indent = 10 \mm }
}

\pageBreak

\score {
  \Cello_b
  \header {
    opus = " "
    piece = \markup { \hspace #44 \abs-fontsize #16 \bold \raise #3 { "II. Menuetto" }}
  }
  \layout { indent = 10 \mm }
}

\pageBreak

\score {
  \Cello_c
  \header {
    opus = " "
    piece = \markup { \hspace #52 \abs-fontsize #16 \bold \raise #2 { "III." }}
  }
  \layout { indent = 10 \mm }
}

\score {
  \Cello_d
  \header {
    opus = " "
    piece = \markup { \hspace #44 \abs-fontsize #16 \bold \raise #3 { "IV. Menuetto" }}
  }
  \layout { indent = 10 \mm }
}

\pageBreak

\score {
  \Cello_e
  \header {
    opus = " "
    piece = \markup { \hspace #53 \abs-fontsize #16 \bold \raise #3 { "V." }}
  }
  \layout { indent = 10 \mm }
}
