```lisp
(define (problem place-tomato-in-bowl)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    tomato - object
    bowl - object
    diningtable - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location tomato diningtable)
    (at-location bowl diningtable)
    (not (holding robot1 tomato))
  )

  (:goal (and
    (at-location tomato bowl)
    (not (holding robot1 tomato))
  ))
)
```