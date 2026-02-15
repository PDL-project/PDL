(define (problem move-tomato-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    tomato - object
    diningtable - object
    countertop - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 diningtable)
    (at-location tomato diningtable)
    (at-location countertop floor)
    (not (holding robot1 tomato))
    (object-close robot1 countertop)
  )

  (:goal (and
    (at-location tomato countertop)
    (not (holding robot1 tomato))
    (object-close robot1 countertop)
  ))

  (:metric minimize (total-cost))
)