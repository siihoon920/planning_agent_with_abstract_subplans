;; ASCII ;;
; W: wall, D: door, k: key, g: gem, G: goal-gem, s: start, .: empty
; old-format y=8..1 (top to bottom in old coords, y=1=bottom)
; converted: new_y = 9 - old_y (y=1=top in new format)
; WWWWWWWW  (new y=1)
; k......W  (new y=2): key1 at x=1
; W.WWWW.W  (new y=3): walls at x=1,3,4,5,6,8
; W.kD.W.W  (new y=4): key2 at x=3, door1 at x=4
; W.WW.W.W  (new y=5): walls at x=1,3,4,6,8
; W.WgDW.W  (new y=6): gem2 at x=4, door2 at x=5
; W.gWg..W  (new y=7): gem1 at x=3, wall at x=4, gem3 at x=5
; WWWWWWWW  (new y=8)
; agent start at (7,3), doors: door1=(4,4), door2=(5,6)
(define (problem doors-keys-gems-11)
  (:domain doors-keys-gems)
  (:objects door1 door2 - door
            key1 key2 - key
            gem1 gem2 gem3 - gem)
  (:init (locked door1)
         (locked door2)
         (= (walls)
            (transpose (bit-mat
               (bit-vec 1 1 1 1 1 1 1 1)
               (bit-vec 0 0 0 0 0 0 0 1)
               (bit-vec 1 0 1 1 1 1 0 1)
               (bit-vec 1 0 0 0 0 1 0 1)
               (bit-vec 1 0 1 1 0 1 0 1)
               (bit-vec 1 0 1 0 0 1 0 1)
               (bit-vec 1 0 0 1 0 0 0 1)
               (bit-vec 1 1 1 1 1 1 1 1))))
         (= (xloc door1) 4)
         (= (yloc door1) 4)
         (= (xloc door2) 5)
         (= (yloc door2) 6)
         (= (xloc key1) 1)
         (= (yloc key1) 2)
         (= (xloc key2) 3)
         (= (yloc key2) 4)
         (= (xloc gem1) 3)
         (= (yloc gem1) 7)
         (= (xloc gem2) 4)
         (= (yloc gem2) 6)
         (= (xloc gem3) 5)
         (= (yloc gem3) 7)
         (= (xpos) 7)
         (= (ypos) 3))
  (:goal (has gem2))
)
