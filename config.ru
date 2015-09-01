use Rack::Static,
    :urls => ["/images", "/css"],
    :root => "public"

map "/" do
    run lambda { |env|
        [
            200,
            {
                'Content-Type' => 'text/html',
                'Cache-Control' => 'public, max-age=86400'
                },
            File.open('public/index.html', File::RDONLY)
            ]
        }
end

map "/contact.html" do
    run lambda { |env|
        [
            200,
            {
                'Content-Type' => 'text/html',
                'Cache-Control' => 'public, max-age=86400'
                },
            File.open('public/contact.html', File::RDONLY)
            ]
        }
end

map "/install.html" do
    run lambda { |env|
        [
            200,
            {
                'Content-Type' => 'text/html',
                'Cache-Control' => 'public, max-age=86400'
                },
            File.open('public/install.html', File::RDONLY)
            ]
        }
end        

map "/userguide.html" do
    run lambda { |env|
        [
            200,
            {
                'Content-Type' => 'text/html',
                'Cache-Control' => 'public, max-age=86400'
                },
            File.open('public/userguide.html', File::RDONLY)
            ]
        }
end
