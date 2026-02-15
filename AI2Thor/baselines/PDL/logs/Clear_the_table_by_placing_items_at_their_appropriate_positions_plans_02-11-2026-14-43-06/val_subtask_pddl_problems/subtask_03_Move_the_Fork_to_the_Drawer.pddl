(define (problem move-pan-to-stoveburner)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    pan - object
    stoveBurner - object
    diningTable - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 diningTable)
    (at-location pan diningTable)
    (at-location stoveBurner floor)
    (not (holding robot1 pan))
  )

  (:goal (and
    (at-location pan stoveBurner)
  ))
)